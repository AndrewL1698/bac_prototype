"""LAION aesthetic predictor V2 (MLP over L2-normalized CLIP ViT-L/14 embeddings). Raw scale ~1-10."""

from __future__ import annotations

import threading

import numpy as np

from montage.log import get_logger

log = get_logger("aesthetic")
_lock = threading.Lock()
_head = None


def _build():
    import torch.nn as nn

    class AestheticMLP(nn.Module):
        def __init__(self, input_size: int = 768):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1024), nn.Dropout(0.2),
                nn.Linear(1024, 128), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.layers(x)

    return AestheticMLP()


def get_head():
    """Loaded once; returns None if weights are unavailable (callers use a neutral score)."""
    global _head
    with _lock:
        if _head is None:
            try:
                import torch
                from montage import weights
                model = _build()
                sd = torch.load(weights.aesthetic_path(), map_location="cpu")
                model.load_state_dict(sd)
                model.eval()
                _head = model
            except Exception as e:  # noqa: BLE001
                log.warning("aesthetic head unavailable (%s); using neutral 5.0", e)
                _head = False
        return _head or None


def score(emb: np.ndarray) -> np.ndarray:
    """(n, 768) normalized embeddings -> (n,) aesthetic scores."""
    emb = np.asarray(emb, np.float32)
    if len(emb) == 0:
        return np.zeros(0, np.float32)
    head = get_head()
    if head is None:
        return np.full(len(emb), 5.0, np.float32)
    import torch
    with torch.no_grad():
        return head(torch.from_numpy(emb)).squeeze(-1).numpy().astype(np.float32)
