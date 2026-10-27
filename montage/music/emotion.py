"""Music emotion (valence/arousal on the 2 Hz grid).

MERT path: frozen MERT-v1-95M features over 5 s windows CENTERED on each grid time, plus the trained
head from DATA_ROOT/weights/emotion_head.pt (phase 9B). `make_windows` and `mert_window_features`
are shared with training (training/emotion/extract_mert.py) so the two cannot drift.
Heuristic path (fallback): arousal from loudness/onsets/brightness, valence 0.
"""

from __future__ import annotations

import threading

import numpy as np

from montage import paths
from montage.config import get_config
from montage.log import get_logger
from montage.music.energy import zscore

log = get_logger("emotion")
_lock = threading.Lock()
_mert = None
_head = None


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
            name = get_config().music.emotion.model
            model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
            fe = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
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
    out = []
    for i in range(0, len(windows), batch):
        x = fe(list(windows[i:i + batch]), sampling_rate=cfg.mert_sr, return_tensors="pt")
        with torch.no_grad():
            hs = model(input_values=x["input_values"].to(dev, dtype), output_hidden_states=True).hidden_states
        out.append(torch.stack(hs, 1).mean(2).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 13, 768), np.float16)


def head_path():
    return paths.weights_dir() / get_config().music.emotion.head_weights


def mert_available() -> bool:
    return bool(get_config().music.emotion.use_mert) and head_path().exists()


def _load_head():
    """Head architecture from spec 09: softmax layer weights -> LayerNorm -> MLP -> tanh -> (valence, arousal)."""
    global _head
    with _lock:
        if _head is None:
            import torch
            import torch.nn as nn

            class EmotionHead(nn.Module):
                def __init__(self, layers: int = 13, dim: int = 768, hidden: int = 256):
                    super().__init__()
                    self.layer_logits = nn.Parameter(torch.zeros(layers))
                    self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                             nn.Linear(hidden, 2), nn.Tanh())

                def forward(self, feats):  # (B, 13, 768)
                    w = torch.softmax(self.layer_logits, 0)
                    return self.mlp((feats * w[None, :, None]).sum(1))

            ckpt = torch.load(head_path(), map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            head = EmotionHead()
            head.load_state_dict(sd)
            _head = head.eval()
        return _head


def heuristic_emotion(curves: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a = np.tanh(0.5 * zscore(curves["rms_db"]) + 0.3 * zscore(curves["onset_density"]) + 0.2 * zscore(curves["centroid"]))
    return a.astype(np.float32), np.zeros_like(a, dtype=np.float32)


def predict_emotion(y24: np.ndarray | None, grid_times: np.ndarray,
                    curves: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """(arousal, valence, source) aligned to grid_times, values in [-1, 1]."""
    if y24 is not None and mert_available():
        try:
            import torch
            feats = mert_window_features(make_windows(y24, grid_times))
            with torch.no_grad():
                va = _load_head()(torch.from_numpy(feats.astype(np.float32))).numpy()
            return np.clip(va[:, 1], -1, 1).astype(np.float32), np.clip(va[:, 0], -1, 1).astype(np.float32), "mert"
        except Exception as e:  # noqa: BLE001  optional component
            log.warning("MERT emotion failed (%s); using heuristic arousal", e)
    if curves is None:
        raise ValueError("heuristic emotion needs energy curves")
    a, v = heuristic_emotion(curves)
    return a, v, "heuristic"
