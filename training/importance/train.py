"""RankNet training for the importance ranker (spec 10 section 4).

Pairs only from the same video: per group per epoch up to 2,000 pairs with |y_i - y_j| >= 0.05; loss
BCEWithLogits(s_i - s_j, [y_i > y_j]). Scalar feature columns are standardized (mean/std saved with the model).
Early stopping on mean Kendall tau over held-out validation videos.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.stats import kendalltau

from montage.importance.ranker import Ranker

N_EMB = 768


def standardize_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Embedding columns stay as they are (mean 0, std 1); scalar columns are standardized."""
    mean = np.zeros(X.shape[1], np.float32)
    std = np.ones(X.shape[1], np.float32)
    mean[N_EMB:] = X[:, N_EMB:].mean(0)
    std[N_EMB:] = np.maximum(X[:, N_EMB:].std(0), 1e-6)
    return mean, std


def sample_pairs(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator, per_group: int = 2000,
                 min_gap: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    I, J = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        a = rng.choice(idx, per_group * 3)
        b = rng.choice(idx, per_group * 3)
        ok = np.abs(y[a] - y[b]) >= min_gap
        I.append(a[ok][:per_group])
        J.append(b[ok][:per_group])
    return np.concatenate(I), np.concatenate(J)


def mean_tau(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    taus = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 3 and np.std(y[m]) > 0:
            t = kendalltau(scores[m], y[m]).statistic
            taus.append(0.0 if np.isnan(t) else t)
    return float(np.mean(taus)) if taus else 0.0


def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, seed: int = 0,
        val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, epochs: int = 30, patience: int = 5) -> Ranker:
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = standardize_stats(X)
    Xs = torch.from_numpy(((X - mean) / std).astype(np.float32))
    model = Ranker.build(kind, X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if kind == "linear" else 3e-4, weight_decay=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, best_state, bad = -1e9, copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        I, J = sample_pairs(y, groups, rng)
        perm = rng.permutation(len(I))
        for k in range(0, len(perm), 1024):
            b = perm[k:k + 1024]
            s = model(Xs[I[b]]).squeeze(-1) - model(Xs[J[b]]).squeeze(-1)
            target = torch.from_numpy((y[I[b]] > y[J[b]]).astype(np.float32))
            loss = lossf(s, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is None:
            continue
        r = Ranker(model.eval(), mean, std, None, kind)
        tau = mean_tau(r.score(val[0]), val[1], val[2])
        if tau > best + 1e-4:
            best, best_state, bad = tau, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if val is not None:
        model.load_state_dict(best_state)
    return Ranker(model.eval(), mean, std, None, kind)
