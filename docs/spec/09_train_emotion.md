# Phase 9B: Music emotion head (MERT + DEAM)

Goal: a small head on frozen MERT-v1-95M features that predicts valence and arousal every 0.5 s, trained and evaluated on DEAM, then used by `music/emotion.py`.

Everything here runs on Colab as background jobs, idempotent per song.

## 1. Data (`training/emotion/download_deam.py`, started in Phase 0)

DEAM: 1,802 songs (1,744 45-second clips from 2013-2014 plus 58 full songs from 2015), MP3, with averaged dynamic valence/arousal annotations at 2 Hz in [-1, 1], starting at 15 s (the first 15 s are not annotated). Annotation CSVs have one row per song and columns like `sample_15000ms, sample_15500ms, ...`. There are separate files for arousal and valence (and std files). Parse generically: find columns matching `sample_(\d+)ms`, read times, drop NaNs. Inspect the unzipped tree first and log what was found.

Note from the DEAM authors: arousal annotations are more reliable than valence.

## 2. Feature extraction (`training/emotion/extract_mert.py`)

For each song (skip if its shard exists):
1. Load mono at 24 kHz.
2. For each annotated time t: window of 5.0 s **centered** at t (zero-pad at edges). This must match `music/emotion.py`.
3. Run MERT with the same processor call as inference (`Wav2Vec2FeatureExtractor`, `sampling_rate=24000`), fp16, batch 64, `output_hidden_states=True`.
4. Per window: stack 13 hidden states, mean over time -> (13, 768), store fp16.
5. Write shard `DATA_ROOT/datasets/deam_mert/<song_id>.npz` with `times`, `feats (n, 13, 768)`, `valence (n,)`, `arousal (n,)`.
Write to scratch and copy each shard to Drive. Report progress per song. Expected: about 125k windows, a few GB, tens of minutes at most on an A100.

Put the shared window/feature code in `montage/music/emotion.py` (functions `make_windows`, `mert_window_features`) and import it here so training and inference cannot drift.

## 3. Splits (`training/emotion/dataset.py`)

- Song-level split, seeded: 80% train / 10% val / 10% test of the 45 s clips. The 58 full songs (2015) are a second test set ("full-songs test").
- A song never appears in two splits.

## 4. Model (`training/emotion/train.py`)

```
w = softmax(layer_logits)                 # 13 learnable scalars
x = sum_l w_l * feats[l]                  # (768,)
h = LayerNorm(x) -> Linear(768, 256) -> GELU -> Dropout(0.2) -> Linear(256, 2) -> tanh
outputs: (valence, arousal)
```
- Loss: MSE + 0.5 * ((1 - CCC_v) + (1 - CCC_a)) computed per batch.
- AdamW lr 1e-3, weight decay 1e-2, batch 512, up to 40 epochs, early stopping on mean val CCC (patience 5). Seeded.
- CCC(x, y) = 2 cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2).

## 5. Baselines and evaluation (`training/emotion/evaluate.py`)

On test and full-songs test, per dimension, pooled over windows and averaged per song:
- **Predict-mean**: training-set mean.
- **Linear probe**: ridge regression on the single MERT layer with the best val score (sweep layers, alpha by val).
- **Head**: the trained model.
Metrics: RMSE, Pearson r, CCC.

Outputs in `DATA_ROOT/eval/emotion/`:
- `report.md` + `metrics.json` (table of the above)
- `layer_weights.png` (bar chart of learned layer weights; good demo visual)
- `curves_<song>.png` for 3 test songs: predicted vs annotated valence and arousal over time

## 6. Deploy

Save `DATA_ROOT/weights/emotion_head.pt` (atomic write; shared with the other person's app): `{state_dict, layer_logits, arch, window_s, align: "centered", metrics, feature_version, created, trained_by, git_commit}`. `music/emotion.py` loads it when present; the app status line then shows `Emotion: MERT`.

Then update the repo docs: the "Music emotion head" table and takeaway in `docs/RESULTS.md`, a `docs/LOG.md` entry with the key numbers and a **Heads-up** that new weights are live, and README "What works today".

Invalidate cached full-song music analyses whose `emotion_source` is heuristic (or include the head's hash in the music cache key) so new runs use the model.

## Acceptance tests

- Extraction is resumable: kill and restart the job; it skips finished shards.
- `test_window_alignment`: training windows and inference windows for the same audio and time are identical arrays.
- Test-set arousal CCC is clearly above predict-mean and at least matches the linear probe; report whatever the numbers are honestly.
- `report.md` and plots exist; the app shows `Emotion: MERT` after deploying; a song's arousal curve visibly rises in loud sections (spot-check plot).
