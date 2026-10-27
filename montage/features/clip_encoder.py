"""CLIP ViT-L/14 (open_clip, OpenAI weights): image and text embeddings, L2-normalized float32."""

from __future__ import annotations

import threading
from typing import Iterable, Sequence

import numpy as np

from montage.config import get_config
from montage.log import get_logger

log = get_logger("clip")
_lock = threading.RLock()
_model = None
FORWARD_COUNT = {"image": 0, "text": 0}  # batches run; tests use this to assert cache hits


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_clip():
    """(model, preprocess, tokenizer), loaded once per process; fp16 on CUDA."""
    global _model
    with _lock:
        if _model is None:
            import open_clip
            import torch
            cfg = get_config().features
            model, _, preprocess = open_clip.create_model_and_transforms(cfg.clip_model, pretrained=cfg.clip_pretrained)
            model.eval().to(device())
            if device() == "cuda":
                model = model.half()
            tok = open_clip.get_tokenizer(cfg.clip_model)
            _model = (model, preprocess, tok)
            log.info("loaded CLIP %s/%s on %s", cfg.clip_model, cfg.clip_pretrained, device())
        return _model


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def encode_images(images: Iterable, batch: int | None = None) -> np.ndarray:
    """images: PIL images or file paths. Returns (n, 768) normalized float32."""
    import torch
    from PIL import Image
    model, preprocess, _ = get_clip()
    batch = batch or get_config().features.clip_batch
    dtype = next(model.parameters()).dtype
    out, buf = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device(), dtype=dtype)
        with torch.no_grad():
            out.append(model.encode_image(x).float().cpu().numpy())
        FORWARD_COUNT["image"] += 1
        buf.clear()

    for im in images:
        if not isinstance(im, Image.Image):
            with Image.open(im) as f:
                im = f.convert("RGB")
        buf.append(preprocess(im.convert("RGB")))
        if len(buf) >= batch:
            flush()
    flush()
    if not out:
        return np.zeros((0, 768), np.float32)
    return _normalize(np.concatenate(out))


_text_cache: dict[str, np.ndarray] = {}


def _encode_text_batch(texts: list[str]) -> np.ndarray:
    import torch
    model, _, tok = get_clip()
    out = []
    for i in range(0, len(texts), 256):
        with torch.no_grad():
            out.append(model.encode_text(tok(texts[i:i + 256]).to(device())).float().cpu().numpy())
        FORWARD_COUNT["text"] += 1
    return _normalize(np.concatenate(out))


def encode_text(texts: Sequence[str]) -> np.ndarray:
    """(n, 768) normalized. Cached per string; uncached strings are encoded in one batch."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 768), np.float32)
    missing = [t for t in dict.fromkeys(texts) if t not in _text_cache]
    if missing:
        with _lock:
            for t, e in zip(missing, _encode_text_batch(missing)):
                _text_cache[t] = e
    return np.stack([_text_cache[t] for t in texts])
