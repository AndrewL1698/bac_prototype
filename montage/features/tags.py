"""Zero-shot tags from the fixed vocabulary (configs/tags.txt)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from montage.config import get_config, repo_path
from montage.features import clip_encoder


@lru_cache(maxsize=4)
def _vocab(path: str) -> tuple[str, ...]:
    lines = repo_path(path).read_text(encoding="utf-8").splitlines()
    return tuple(t.strip() for t in lines if t.strip() and not t.strip().startswith("#"))


def vocabulary() -> list[str]:
    return list(_vocab(get_config().features.tags_file))


def tag_embeddings() -> np.ndarray:
    return clip_encoder.encode_text([f"a photo of {t}" for t in vocabulary()])


def top_tags(emb: np.ndarray, ids: list[str]) -> dict[str, list[str]]:
    cfg = get_config().features
    vocab = vocabulary()
    sims = emb @ tag_embeddings().T
    out = {}
    for cid, row in zip(ids, sims):
        order = np.argsort(-row)[: cfg.tags_top_k]
        out[cid] = [vocab[j] for j in order if row[j] >= cfg.tag_min_cos]
    return out
