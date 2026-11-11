"""Music emotion (valence/arousal on the 2 Hz grid).

MERT path: frozen MERT-v1-95M features over 5 s windows CENTERED on each grid time, then a trained head:
- temporal head (phase 14, DATA_ROOT/weights/emotion_temporal.pt): a BiGRU over the whole song's window
  sequence, so each prediction sees its neighbours;
- window head (phase 9B, DATA_ROOT/weights/emotion_head.pt): each window on its own.
The temporal head is used when enabled (music.emotion.use_temporal) and present; if it is missing or fails to load,
the window head; then the heuristic.
`make_windows`, `mert_window_features`, `make_head` and `make_temporal_head` are shared with training
(training/emotion/) so the two cannot drift.
Heuristic path (fallback): arousal from loudness/onsets/brightness, valence 0.
"""

from __future__ import annotations

import threading

import numpy as np

from montage import cache, paths
from montage.config import get_config
from montage.log import get_logger
from montage.music.energy import zscore

log = get_logger("emotion")
_lock = threading.Lock()
_mert = None
_head = None
_temporal = None   # the loaded head, or False once loading failed (window head for the rest of the process)


def make_windows(y24: np.ndarray, times: np.ndarray, sr: int | None = None, window_s: float | None = None) -> np.ndarray:
    """(n, window_len) float32 windows centered on `times` (seconds), zero-padded at the edges."""
    cfg = get_config().music
    sr = sr or cfg.mert_sr
    n = int(round((window_s or cfg.emotion.window_s) * sr))
    half = n // 2
    pad = np.pad(np.asarray(y24, np.float32), (half, n))
    out = np.zeros((len(times), n), np.float32)
    for i, t in enumerate(times):
        c = int(round(t * sr)) + half  # center index in padded signal
        out[i] = pad[c - half:c - half + n]
    return out


def get_mert():
    """(model, feature_extractor), loaded once; fp16 on CUDA."""
    global _mert
    with _lock:
        if _mert is None:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            cfg = get_config().music.emotion
            kw = {"trust_remote_code": True, "revision": cfg.revision} if cfg.revision else {"trust_remote_code": True}
            model = AutoModel.from_pretrained(cfg.model, **kw).eval()
            fe = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model, **kw)
            if torch.cuda.is_available():
                model = model.half().cuda()
            _mert = (model, fe)
        return _mert


def mert_window_features(windows: np.ndarray, batch: int | None = None) -> np.ndarray:
    """(n, 13, 768) float16: all hidden states, mean over time, per window."""
    import torch
    model, fe = get_mert()
    cfg = get_config().music
    batch = batch or cfg.emotion.batch
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # Newer transformers can drop `hidden_states` for remote-code models (MERT returns None under 5.20).
    # Hooks record the same 13 tensors: the encoder input (input to layer 0) plus each layer's output.
    caps: list = []
    layers = model.encoder.layers
    hooks = [layers[0].register_forward_pre_hook(lambda mod, args: caps.append(args[0]))]
    hooks += [layer.register_forward_hook(lambda mod, inp, o: caps.append(o[0] if isinstance(o, tuple) else o))
              for layer in layers]
    out = []
    try:
        for i in range(0, len(windows), batch):
            x = fe(list(windows[i:i + batch]), sampling_rate=cfg.mert_sr, return_tensors="pt")
            caps.clear()
            with torch.no_grad():
                hs = model(input_values=x["input_values"].to(dev, dtype), output_hidden_states=True).hidden_states
            if hs is None:
                if len(caps) != len(layers) + 1:
                    raise RuntimeError(f"captured {len(caps)} MERT layer outputs, expected {len(layers) + 1}")
                hs = caps
            out.append(torch.stack(list(hs), 1).mean(2).float().cpu().numpy().astype(np.float16))
    finally:
        for h in hooks:
            h.remove()
    return np.concatenate(out) if out else np.zeros((0, 13, 768), np.float16)


def head_path():
    return paths.weights_dir() / get_config().music.emotion.head_weights


def temporal_path():
    return paths.weights_dir() / get_config().music.emotion.temporal_weights


def temporal_available() -> bool:
    cfg = get_config().music.emotion
    return bool(cfg.use_mert and cfg.use_temporal) and _temporal is not False and temporal_path().exists()


def mert_available() -> bool:
    return bool(get_config().music.emotion.use_mert) and (head_path().exists() or temporal_available())


def emotion_model_id() -> str:
    """Which emotion model a new analysis would use, with its weights hash: 'temporal:<h>', 'window:<h>' or
    'heuristic'. Cached song analyses store it, so retraining or deploying a head invalidates them."""
    if temporal_available():
        return f"temporal:{cache.file_hash(temporal_path())}"
    if mert_available():
        return f"window:{cache.file_hash(head_path())}"
    return "heuristic"


def make_head(layers: int = 13, dim: int = 768, hidden: int = 256):
    """Spec 09 head: softmax layer weights -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> tanh.
    Outputs (valence, arousal). Shared by training/emotion/train.py and inference."""
    import torch
    import torch.nn as nn

    class EmotionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_logits = nn.Parameter(torch.zeros(layers))
            self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(hidden, 2), nn.Tanh())

        def forward(self, feats):  # (B, 13, 768)
            w = torch.softmax(self.layer_logits, 0)
            return self.mlp((feats * w[None, :, None]).sum(1))

    return EmotionHead()


def make_temporal_head(layers: int = 13, dim: int = 768, proj_dim: int = 128, hidden: int = 64,
                       dropout: float = 0.2):
    """Spec 14 head. Per window: softmax layer weights -> LayerNorm -> Linear(dim, proj_dim) -> GELU -> Dropout.
    Across windows: a 1-layer bidirectional GRU. Per window again: [projection, GRU state] -> Dropout -> Linear -> tanh.
    The skip from the projection lets it fall back to per-window behaviour. Input (B, T, layers, dim) plus
    optional lengths (B,) for padded batches; output (B, T, 2) = (valence, arousal).
    Shared by training/emotion/temporal.py and inference."""
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class TemporalEmotionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_logits = nn.Parameter(torch.zeros(layers))
            self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
            self.gru = nn.GRU(proj_dim, hidden, batch_first=True, bidirectional=True)
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(proj_dim + 2 * hidden, 2), nn.Tanh())

        def forward(self, feats, lengths=None):  # (B, T, 13, 768)
            w = torch.softmax(self.layer_logits, 0)
            x = self.proj((feats * w[None, None, :, None]).sum(2))
            if lengths is None:
                h, _ = self.gru(x)
            else:  # packing keeps padding from leaking into the backward direction
                packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                h, _ = self.gru(packed)
                h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
            return self.out(torch.cat([x, h], -1))

    return TemporalEmotionHead()


def _load_temporal():
    global _temporal
    with _lock:
        if _temporal is None:
            import torch
            try:
                ckpt = torch.load(temporal_path(), map_location="cpu", weights_only=False)
                head = make_temporal_head(**ckpt["arch"])
                head.load_state_dict(ckpt["state_dict"])
            except Exception:
                _temporal = False   # emotion_model_id() then names the window head, so the cache stays valid
                raise
            _temporal = head.eval()
        return _temporal


def _load_head():
    global _head
    with _lock:
        if _head is None:
            import torch
            ckpt = torch.load(head_path(), map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            head = make_head()
            head.load_state_dict(sd)
            _head = head.eval()
        return _head


def heuristic_emotion(curves: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a = np.tanh(0.5 * zscore(curves["rms_db"]) + 0.3 * zscore(curves["onset_density"]) + 0.2 * zscore(curves["centroid"]))
    return a.astype(np.float32), np.zeros_like(a, dtype=np.float32)


def _av(va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) valence/arousal -> (arousal, valence) clipped to [-1, 1]."""
    return np.clip(va[:, 1], -1, 1).astype(np.float32), np.clip(va[:, 0], -1, 1).astype(np.float32)


def predict_emotion_detail(y24: np.ndarray | None, grid_times: np.ndarray,
                           curves: dict[str, np.ndarray] | None = None
                           ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """(arousal, valence, source, model_id) aligned to grid_times, values in [-1, 1].
    source is 'mert' or 'heuristic' (the MusicProfile contract); model_id says which head actually ran."""
    if y24 is not None and mert_available():
        feats = None
        try:
            feats = mert_window_features(make_windows(y24, grid_times))
        except Exception as e:  # noqa: BLE001  optional component
            log.warning("MERT features failed (%s); using heuristic arousal", e)
        if feats is not None and temporal_available():
            try:
                import torch
                with torch.no_grad():
                    va = _load_temporal()(torch.from_numpy(feats.astype(np.float32))[None])[0].numpy()
                return (*_av(va), "mert", f"temporal:{cache.file_hash(temporal_path())}")
            except Exception as e:  # noqa: BLE001  optional component
                log.warning("temporal emotion head failed (%s); trying the window head", e)
        if feats is not None and head_path().exists():
            try:
                import torch
                with torch.no_grad():
                    va = _load_head()(torch.from_numpy(feats.astype(np.float32))).numpy()
                return (*_av(va), "mert", f"window:{cache.file_hash(head_path())}")
            except Exception as e:  # noqa: BLE001  optional component
                log.warning("MERT emotion head failed (%s); using heuristic arousal", e)
    if curves is None:
        raise ValueError("heuristic emotion needs energy curves")
    a, v = heuristic_emotion(curves)
    return a, v, "heuristic", "heuristic"


def predict_emotion(y24: np.ndarray | None, grid_times: np.ndarray,
                    curves: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """(arousal, valence, source) aligned to grid_times, values in [-1, 1]."""
    a, v, src, _ = predict_emotion_detail(y24, grid_times, curves)
    return a, v, src
