# Importance scoring

Implemented in two steps: the **heuristic** in Phase 3 (Day 1), the **learned ranker blend** in Phase 10A (Day 2). The planner only sees `ImportanceScores.final` in [0, 1].

## 1. Shared feature vector (`importance/ranker.py: ranker_features(arrays) -> (N, D)`)

Used by both training (spec 10) and inference. Exactly:
`[emb (768), aesthetic, sharpness, brightness, contrast, exposure_clip, colorfulness, min(face_count, 3), face_area]`
No motion or shake (photos have none; motion enters through the heuristic). Scalar columns are standardized with the mean/std saved alongside the trained weights.

## 2. Heuristic (`importance/heuristic.py`)

All z-scores are within the current upload set. Videos and photos are z-scored together except motion/shake (videos only; photos get 0 for those terms).

```
H = w_aes*z(aesthetic) + w_sharp*z(sharpness) - w_exp*z(exposure_clip)
    - w_bright*|brightness - 0.5|*4 + w_face_count*min(face_count,3)/3 + w_face_area*z(face_area)
    + w_color*z(colorfulness) + w_motion*z(motion) - w_shake*z(shake) - w_junk*is_junk
```
Defaults in config (`importance.heuristic`): aes 1.0, sharp 0.6, exp 0.4, bright 0.3, face_count 0.5, face_area 0.3, color 0.3, motion 0.4, shake 0.5, junk 3.0.
`is_junk` = candidate has any tag in `importance.junk_tags` (screenshot, text document, receipt, meme).

**Hard rejects** (excluded from planning unless pinned by the director): sharpness below the batch 10th percentile AND below `importance.abs_min_sharpness`; or exposure_clip > 0.5. If rejects would leave fewer candidates than slots, un-reject the best of them.

## 3. Learned ranker (inference)

`Ranker.load(DATA_ROOT/weights/importance_ranker.pt)` returns an object with `score(X) -> (N,)`, or None if missing/failed (log fallback `importance:heuristic`). Model types supported: `linear` and `mlp` (see spec 10); the checkpoint stores type, weights, feature mean/std, and validation metrics.

## 4. Blend (`importance/blend.py`)

```
r_H = rank_normalize(H, within modality)        # photos ranked among photos, videos among videos, to [0,1]
r_L = rank_normalize(L, within modality)         # if ranker available
final = alpha * r_L + (1 - alpha) * r_H          # alpha = importance.blend_alpha (default 0.5)
final = r_H                                      # if no ranker
```
`rank_normalize`: average ranks scaled to [0, 1]; a modality with a single item gets 0.5.

Rationale: annotator importance is only meaningful within a collection, and the planner needs relative quality inside one upload set. Normalizing per modality stops one modality from dominating because the model scores it systematically higher; the mix of photos and videos is controlled separately by the planner.

## Acceptance tests

- `test_heuristic_prefers_sharp`: among fixture photos, the blurry one ranks in the bottom 2.
- `test_junk_penalty`: a candidate tagged screenshot drops below the median.
- `test_rank_normalize_modality`: photos-only and videos-only each span [0, 1].
- `test_blend_without_ranker`: with no weights file, `final == r_H` and `source == "heuristic"`.
- `test_reject_relaxation`: if rejects exceed the allowance, best rejected candidates come back.
