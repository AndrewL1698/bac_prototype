"""Train the emotion head on DEAM MERT features, evaluate, and deploy (spec 09 sections 4-6).

    python scripts/run_job.py --name emo_train -- python -m training.emotion.train

Writes DATA_ROOT/eval/emotion/{report.md, metrics.json, layer_weights.png, curves_<song>.png} and
DATA_ROOT/weights/emotion_head.pt (atomic). The app picks the head up automatically (Emotion: MERT).
"""

from __future__ import annotations

import copy
import io
from datetime import datetime

import numpy as np

from montage import jobs, paths
from montage.config import get_config
from montage.jobs import git_info
from montage.log import get_logger
from montage.music.emotion import head_path, make_head
from training.emotion import evaluate
from training.emotion.dataset import load_all
from training.emotion.extract_mert import shard_dir

log = get_logger("emo_train")


def _ccc_t(x, y):
    import torch
    xm, ym = x.mean(), y.mean()
    cov = ((x - xm) * (y - ym)).mean()
    return 2 * cov / (x.var(unbiased=False) + y.var(unbiased=False) + (xm - ym) ** 2 + 1e-8)


def predict(head, feats: np.ndarray, batch: int = 4096) -> np.ndarray:
    import torch
    dev = next(head.parameters()).device
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            x = torch.from_numpy(feats[i:i + batch].astype(np.float32)).to(dev)
            out.append(head(x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)


def train(splits, seed: int = 0, epochs: int = 40, patience: int = 5, batch: int = 512):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = splits["train"]
    X = torch.from_numpy(tr.feats).to(dev)                     # fp16 on the GPU (~2 GB)
    Y = torch.from_numpy(tr.y).to(dev)
    head = make_head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad, hist = -1e9, None, 0, []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(X), generator=g).to(dev)
        tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            p = head(X[idx].float())
            y = Y[idx]
            loss = torch.nn.functional.mse_loss(p, y) + 0.5 * ((1 - _ccc_t(p[:, 0], y[:, 0])) + (1 - _ccc_t(p[:, 1], y[:, 1])))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        pv = predict(head, splits["val"].feats)
        val = evaluate.mean_ccc(pv, splits["val"].y)
        hist.append({"epoch": ep, "train_loss": tot / len(X), "val_mean_ccc": val})
        log.info("epoch %d loss %.4f val mean CCC %.4f", ep, tot / len(X), val)
        jobs.report(0.1 + 0.6 * (ep + 1) / epochs, f"epoch {ep + 1}: val CCC {val:.3f}")
        if val > best + 1e-4:
            best, best_state, bad = val, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return head.cpu(), hist, best


def main() -> int:
    import torch
    paths.load_colab_env()
    cfg = get_config()
    jobs.report(0.02, "loading MERT shards", force=True)
    ids, splits = load_all(shard_dir(), seed=cfg.seed)
    log.info("songs: %s", {k: len(v) for k, v in ids.items()})
    head, hist, best = train(splits, seed=cfg.seed)
    jobs.report(0.75, "evaluating", force=True)
    metrics = evaluate.run(head, splits, ids, hist)
    _, commit = git_info()
    ckpt = {"state_dict": head.state_dict(), "layer_logits": head.layer_logits.detach().numpy().tolist(),
            "arch": "softmax(13 layers) -> LayerNorm -> Linear(768,256) -> GELU -> Dropout(0.2) -> Linear(256,2) -> tanh",
            "outputs": ["valence", "arousal"], "window_s": cfg.music.emotion.window_s, "align": "centered",
            "metrics": metrics, "feature_version": f"{cfg.music.emotion.model}@{cfg.music.emotion.revision}",
            "created": datetime.now().isoformat(timespec="seconds"), "trained_by": paths.user(), "git_commit": commit}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    paths.atomic_write(head_path(), buf.getvalue())
    t = metrics["test"]["head"]
    print(f"deployed {head_path()}: test CCC valence {t['valence']['ccc']:.3f}, arousal {t['arousal']['ccc']:.3f}; "
          f"best val mean CCC {best:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
