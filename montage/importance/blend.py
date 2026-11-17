"""Final importance: rank-normalize within modality (photos, videos), blend the heuristic with that modality's
learned ranker if present."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from montage import paths
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


def importance(features, ranker: Ranker | None | bool = None,
               photo_ranker: Ranker | None | bool = None) -> ImportanceScores:
    """Photos and videos are ranked separately, each with its own learned ranker blended with the heuristic.

    ranker scores videos (and photos when there is no photo ranker); photo_ranker scores photos. None loads the
    checkpoint if present; ranker=False forces heuristic only for everything (ablation); photo_ranker=False makes
    photos use the video ranker."""
    arr, ids = features.arrays, features.ids
    is_video = np.asarray(arr["is_video"], bool)
    groups = np.where(is_video, 1, 0)
    H = heuristic_scores(arr, features.table.tags, ids)
    r_h = rank_normalize(H, groups)
    if ranker is False:
        return ImportanceScores(candidate_ids=ids, heuristic=H.tolist(), learned=None, final=r_h.tolist(),
                                source="heuristic")
    if ranker is None:
        ranker = Ranker.load()
    if photo_ranker is None:
        photo_ranker = Ranker.load(paths.weights_dir() / get_config().importance.photo_ranker_weights)
    by_modality = ((False, photo_ranker or ranker), (True, ranker))
    if not any(r for _, r in by_modality):
        return ImportanceScores(candidate_ids=ids, heuristic=H.tolist(), learned=None, final=r_h.tolist(),
                                source="heuristic")
    X = ranker_features(arr)
    L = np.zeros(len(ids))   # stays 0 for a modality without a ranker (its final score is the heuristic's)
    final = r_h.copy()
    for video, r in by_modality:
        m = is_video == video
        if r and m.any():
            L[m] = r.score(X[m])
            alpha = r.blend_alpha if r.blend_alpha is not None else get_config().importance.blend_alpha
            final[m] = alpha * rank_normalize(L[m], groups[m]) + (1 - alpha) * r_h[m]
    return ImportanceScores(candidate_ids=ids, heuristic=H.tolist(), learned=L.tolist(), final=final.tolist(),
                            source="blend")
