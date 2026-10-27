"""Learned importance ranker (phase 10A): shared feature vector + inference loader.

Checkpoint DATA_ROOT/weights/importance_ranker.pt (spec 10):
    {type: "linear"|"mlp", state_dict, feat_mean, feat_std, blend_alpha, metrics, feature_version, ...}
"""

from __future__ import annotations

import numpy as np

from montage import paths
from montage.config import get_config
from montage.log import get_logger

log = get_logger("ranker")
SCALAR_KEYS = ("aesthetic", "sharpness", "brightness", "contrast", "exposure_clip", "colorfulness")


def ranker_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """(N, 768 + 8): [emb, aesthetic, sharpness, brightness, contrast, exposure_clip, colorfulness,
    min(face_count, 3), face_area]. Identical for training and inference."""
    cols = [np.asarray(arrays[k], np.float32)[:, None] for k in SCALAR_KEYS]
    cols.append(np.minimum(np.asarray(arrays["face_count"], np.float32), 3)[:, None])
    cols.append(np.asarray(arrays["face_area"], np.float32)[:, None])
    return np.concatenate([np.asarray(arrays["emb"], np.float32)] + cols, axis=1)


class Ranker:
    def __init__(self, model, mean: np.ndarray, std: np.ndarray, blend_alpha: float | None, kind: str):
        self.model, self.mean, self.std, self.blend_alpha, self.kind = model, mean, std, blend_alpha, kind

    @staticmethod
    def build(kind: str, dim: int):
        import torch.nn as nn
        if kind == "linear":
            return nn.Linear(dim, 1)
        return nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

    @classmethod
    def load(cls, path=None) -> "Ranker | None":
        path = path or paths.weights_dir() / get_config().importance.ranker_weights
        if not path.exists():
            return None
        try:
            import torch
            ck = torch.load(path, map_location="cpu", weights_only=False)
            mean = np.asarray(ck["feat_mean"], np.float32)
            std = np.asarray(ck["feat_std"], np.float32)
            model = cls.build(ck["type"], len(mean))
            model.load_state_dict(ck["state_dict"])
            model.eval()
            return cls(model, mean, std, ck.get("blend_alpha"), ck["type"])
        except Exception as e:  # noqa: BLE001  optional component
            log.warning("importance ranker unavailable (%s); heuristic only", e)
            return None

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch
        Xs = (X - self.mean) / np.where(self.std > 1e-9, self.std, 1.0)
        with torch.no_grad():
            return self.model(torch.from_numpy(Xs.astype(np.float32))).squeeze(-1).numpy()
