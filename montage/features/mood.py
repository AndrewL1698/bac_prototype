"""Visual mood: CLIP prompt-ensemble poles (raw, cacheable) combined per batch into arousal/valence."""

from __future__ import annotations

import numpy as np

from montage.config import get_config
from montage.features import clip_encoder


def _zs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def pole_embeddings() -> dict[str, np.ndarray]:
    cfg = get_config().mood
    out = {}
    for pole, phrases in cfg.prompts.model_dump().items():
        texts = [t.format(p) for p in phrases for t in cfg.templates]
        e = clip_encoder.encode_text(texts).mean(0)
        out[pole] = e / np.linalg.norm(e)
    return out


def mood_raw(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (arousal_raw, valence_raw) = 100 * (cos(e, high) - cos(e, low))."""
    p = pole_embeddings()
    a = 100.0 * (emb @ p["arousal_high"] - emb @ p["arousal_low"])
    v = 100.0 * (emb @ p["valence_pos"] - emb @ p["valence_neg"])
    return a.astype(np.float32), v.astype(np.float32)


def combine(arr: dict[str, np.ndarray], is_video: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batch-relative visual arousal/valence (z-scored over the current upload set)."""
    cfg = get_config().mood
    wa, wv = cfg.arousal_weights, cfg.valence_weights
    z_motion = np.full(len(is_video), cfg.photo_motion_z)
    if is_video.any():
        z_motion[is_video] = _zs(arr["motion"][is_video])
    va = wa["clip"] * _zs(arr["mood_arousal_raw"]) + wa["motion"] * z_motion + wa["color"] * _zs(arr["colorfulness"])
    vv = wv["clip"] * _zs(arr["mood_valence_raw"]) + wv["brightness"] * _zs(arr["brightness"]) \
        + wv["color"] * _zs(arr["colorfulness"])
    return _zs(va).astype(np.float32), _zs(vv).astype(np.float32)
