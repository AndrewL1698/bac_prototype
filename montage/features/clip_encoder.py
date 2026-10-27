"""CLIP ViT-L/14 (open_clip, OpenAI weights): image and text embeddings, L2-normalized float32."""

from __future__ import annotations

import threading
from typing import Sequence

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
            # OpenAI weights were trained with QuickGELU; open_clip's plain ViT-L-14 config omits it.
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.clip_model, pretrained=cfg.clip_pretrained, force_quick_gelu=cfg.clip_pretrained == "openai")
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


def _prep(item):
    from PIL import Image
    _, preprocess, _ = get_clip()
    if isinstance(item, Image.Image):
        return preprocess(item.convert("RGB"))
    with Image.open(item) as f:
        return preprocess(f.convert("RGB"))


def encode_images(images: Sequence, batch: int | None = None, workers: int = 8) -> np.ndarray:
    """images: PIL images or file paths. Returns (n, 768) normalized float32.
    Preprocessing (decode + resize) runs in a thread pool; the GPU sees full batches."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    model, _, _ = get_clip()
    images = list(images)
    if not images:
        return np.zeros((0, 768), np.float32)
    batch = batch or get_config().features.clip_batch
    dtype = next(model.parameters()).dtype
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(images), batch):
            x = torch.stack(list(ex.map(_prep, images[i:i + batch]))).to(device(), dtype=dtype)
            with torch.no_grad():
                out.append(model.encode_image(x).float().cpu().numpy())
            FORWARD_COUNT["image"] += 1
    return _normalize(np.concatenate(out))


# Text embeddings: in-memory dict backed by one shared npz on Drive, so a warm run
# never needs the CLIP text tower for strings it has seen before (tags, mood prompts, queries).
_text_cache: dict[str, np.ndarray] = {}
_text_disk_loaded = False


def _text_disk_path():
    from montage import paths
    cfg = get_config().features
    return paths.cache_dir("text") / f"{cfg.clip_model}_{cfg.clip_pretrained}.npz"


def _load_text_disk() -> None:
    global _text_disk_loaded
    if _text_disk_loaded:
        return
    _text_disk_loaded = True
    from montage import cache
    z = cache.load_npz(_text_disk_path())
    if z and "texts" in z:
        for t, e in zip(z["texts"].tolist(), z["emb"]):
            _text_cache.setdefault(str(t), e.astype(np.float32))


def _save_text_disk() -> None:
    from montage import cache
    texts = list(_text_cache)
    try:
        cache.save_npz(_text_disk_path(), texts=np.array(texts), emb=np.stack([_text_cache[t] for t in texts]))
    except OSError as e:
        log.warning("could not save text cache: %s", e)


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
    """(n, 768) normalized. Cached per string (memory + Drive); misses are encoded in one batch."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 768), np.float32)
    with _lock:
        _load_text_disk()
        missing = [t for t in dict.fromkeys(texts) if t not in _text_cache]
        if missing:
            for t, e in zip(missing, _encode_text_batch(missing)):
                _text_cache[t] = e
            _save_text_disk()
        return np.stack([_text_cache[t] for t in texts])
