"""Final importance: rank-normalize within modality, blend heuristic with the learned ranker if present."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from montage.config import get_config
from montage.importance.heuristic import heuristic_scores
from montage.importance.ranker import Ranker, ranker_features
from montage.types import ImportanceScores


def rank_normalize(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Average ranks scaled to [0, 1] within each group; a group with one item gets 0.5."""
    x = np.asarray(x, float)
    out = np.zeros(len(x))
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        out[m] = 0.5 if n == 1 else (rankdata(x[m]) - 1) / (n - 1)
    return out


def importance(features, ranker: Ranker | None | bool = None) -> ImportanceScores:
    """ranker=None loads the checkpoint if present; ranker=False forces heuristic only (ablation)."""
    arr, ids = features.arrays, features.ids
    groups = np.where(np.asarray(arr["is_video"], bool), 1, 0)
    H = heuristic_scores(arr, features.table.tags, ids)
    r_h = rank_normalize(H, groups)
    if ranker is None:
        ranker = Ranker.load()
    if ranker:
        L = ranker.score(ranker_features(arr))
        alpha = ranker.blend_alpha if ranker.blend_alpha is not None else get_config().importance.blend_alpha
        final = alpha * rank_normalize(L, groups) + (1 - alpha) * r_h
        return ImportanceScores(candidate_ids=ids, heuristic=H.tolist(), learned=L.tolist(),
                                final=final.tolist(), source="blend")
    return ImportanceScores(candidate_ids=ids, heuristic=H.tolist(), learned=None, final=r_h.tolist(),
                            source="heuristic")
