"""Near-duplicate clusters over candidate embeddings (computed per batch, never cached)."""

from __future__ import annotations

import numpy as np

from montage.config import get_config


def dup_clusters(emb: np.ndarray, asset_ids: list[str], kinds: list[str]) -> np.ndarray:
    """Average-linkage cosine clustering at 1 - dedup.cos_threshold.

    Windows of the same video never share a cluster (they overlap by design; the planner limits
    same-source picks separately): within a cluster, the first window of each video keeps the
    cluster and its other windows become singletons.
    """
    n = len(emb)
    if n == 0:
        return np.zeros(0, int)
    if n == 1:
        return np.zeros(1, int)
    from sklearn.cluster import AgglomerativeClustering
    thr = 1.0 - get_config().dedup.cos_threshold
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=thr).fit_predict(emb).astype(int)
    next_id = labels.max() + 1
    for lab in np.unique(labels):
        seen: set[str] = set()
        for i in np.where(labels == lab)[0]:
            if kinds[i] != "video":
                continue
            if asset_ids[i] in seen:
                labels[i] = next_id
                next_id += 1
            else:
                seen.add(asset_ids[i])
    # relabel to 0..k-1 in order of first appearance
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(int(x), len(remap)) for x in labels], int)
