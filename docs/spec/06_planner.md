# Phase 5: Slots, costs, beam search, EDL

Goal: decide when every cut happens (slots), then choose and order shots for those slots by minimizing a cost, and write the EDL. This is the core of the project; keep it readable and fully configurable.

Notation: candidates i (after excludes/rejects), slots j = 0..K-1. Lower cost is better. All weights live under `planner.*` in config.

## 1. Slot grid (`planning/slots.py: build_slots(profile, directives, n_usable) -> list[Slot]`)

Beat duration d = local inter-beat interval.

1. **Base length from intensity.** At the current beat, let z = mean intensity over the next beat. `level = clip((z + 1) / 2.5, 0, 1)`; `beats_cont = 8 * 2^(-3*level)` (level 0 -> 8 beats, level 1 -> 1 beat).
2. **Pacing.** Multiply by `2^(-p)` where p is the section's effective pacing (spec 05 section 3).
3. **Snap** to the nearest of {1, 2, 4, 8} in log2 space. Enforce `planner.min_shot_s` (0.5) and `planner.max_shot_s` (4.0) by doubling/halving.
4. **Walk** from beat 0 (excerpt starts on a downbeat): next cut at beat k + n.
5. **Forced cuts** at section boundaries (already on downbeats): if the next cut would pass a boundary, cut at the boundary instead. If that shot would be shorter than `min_shot_s`, move the previous cut to the boundary instead (merge). Mark the slot starting at a boundary `forced=True`.
6. The last slot ends exactly at the excerpt end.
7. **Scarcity.** n_usable counts the shots that can appear together without repeats (`planning.distinct_shots`): each `dup_cluster` at most `max_per_cluster` times, each video at most its non-overlapping windows (`max_windows_per_video`). Near-duplicate bursts and overlapping video windows are not spare footage. If K > floor(`planner.max_fill_ratio` (0.9) * n_usable): double the beats of the lowest-intensity sections first and rebuild; repeat. If K still exceeds n_usable at max lengths, set `reuse_mode=True` (beam allows reuse; EDL note).
8. Per slot: `intensity`, `arousal`, `valence` = means over the slot from the grid; `section_idx`; `on_downbeat` (start within half a frame of a downbeat); `boundary_strength` = 1.0 if forced else 0.5 * novelty at the start time (0 for slot 0).

## 2. Precomputed cost terms (`planning/costs.py`)

Precompute matrices once per plan (numpy):

**Ranks** (over usable candidates): `vr_i` = rank-normalized visual arousal; `vv_i` = rank-normalized visual valence. **Slot targets**: `ta_j` from directives (target arousal rank); `tv_j` = rank of slot valence among slots.

**Unary U (N x K):**
```
U(i,j) = - w_imp * imp_i
         + w_mood * |vr_i - ta_j|
         + w_val  * |vv_i - tv_j|        # w_val = 0 when emotion_source == "heuristic"
         - w_dir  * D(i,j)               # director score, spec 05
         + INF  if video and (src_max_end - src_start) < duration_j      # infeasible
         + w_photo_short   if photo and duration_j < photo_min_s
         + w_photo_hi * irank_j  if photo and videos exist               # irank_j: slot intensity rank
         - w_must          if i is must-include or pinned
```
Defaults: w_imp 1.0, w_mood 0.8, w_val 0.3, w_dir 1.0, photo_min_s 0.6, w_photo_short 0.5, w_photo_hi 0.3, w_must 5.0.

**Variety noise:** `U += planner.variety * std(U_finite) * Gumbel(seed)` (default variety 0.15). Same seed gives the same edit.

**Pairwise P (N x N) plus boundary-dependent parts**, for i followed by i' where i' starts slot j:
```
P = w_rep  * relu(cos(e_i, e_i') - rep_tau)
  + w_src  * [asset(i) == asset(i')]
  + w_jump * vj(i,i') * (1 - b_j)  -  w_contrast * vj(i,i') * b_j
  + w_chrono * [taken_at(i') < taken_at(i) - chrono_tol_s]     # only if plan.chronological and both timestamps known
```
`vj` = visual jump: 0.5 * chi-square distance of 8x8x8 RGB histograms (of thumbnails) + 0.5 * 2*|mean luma difference|, then divided by the batch 95th percentile and clipped to [0, 1].
Defaults: w_rep 2.0, rep_tau 0.75, w_src 1.5, w_jump 0.6, w_contrast 0.6, w_chrono 1.0, chrono_tol_s 60.

## 3. Constraints

- Each candidate at most once (unless `reuse_mode`: then every repeat of a shot or of its `dup_cluster` costs `w_reuse` 2.0 per earlier use, and two shots of one cluster are never adjacent).
- Each `dup_cluster` at most `planner.max_per_cluster` (1).
- Each video asset at most `planner.max_windows_per_video` (3) shots, and chosen windows of one asset must not overlap in source time.
- Photo fraction: soft target band [`min_photo_frac`, `max_photo_frac`] (0.3, 0.7), shrunk to what is available. Penalty `w_ratio * dist_outside_band^2 * K` added when scoring complete sequences; states already above the max pay the incremental share.
- Must-include assets: a complete sequence missing any feasible one is invalid. Prune states where remaining slots < remaining required items.

## 4. Beam search (`planning/beam.py`)

1. **Pool:** usable candidates ranked by `imp_i + 0.5 * max_j D(i,j)`; keep the top `M = max(3K, planner.min_pool)` (min_pool 60) plus all must-include/pinned.
2. **State:** cumulative cost, sequence, used set (bitset), cluster counts, per-asset chosen source intervals, photo count, required remaining.
3. **Init:** slot 0, every feasible candidate.
4. **Expand:** for each state, compute costs for all candidates at once (vectorized: `cost + U[:, j] + P[last, :]` with masks for constraints), take the best `planner.expand_top` (32) per state, then keep the global best `planner.beam_width` (128) states.
5. **Finish:** add end penalties, discard invalid, return the min-cost sequence and its cost breakdown per shot.
6. If no valid sequence exists: retry with `reuse_mode=True`; if still none, raise `UserError("Not enough usable footage for this song length. Add more media or shorten the video.")`.

Performance target: K=40, M=120, B=128 in under 2 s on CPU.

## 5. In/out points (`planning/inout.py`)

For each chosen video candidate and slot duration dur:
- Search `src_in` in 0.1 s steps over [src_start, src_max_end - dur].
- Score: mean over analysis frames inside [src_in, src_in + dur] of `0.4*z(frame aesthetic) + 0.4*z(frame motion) + 0.2*z(frame director relevance)` (relevance to the slot's segment queries, else vibe queries), minus `0.1 * (src_in - src_start)` to stay near the window.
- Must not overlap other chosen intervals from the same asset.

## 6. EDL (`planning/edl.py`)

- **Frame quantization:** cut times t_0=0 < t_1 < ... < t_K = L. Frame indices f_k = round(t_k * fps), computed for all cuts at once. Shot j covers frames [f_j, f_{j+1}). Never round durations per shot.
- `framing` and `kenburns` from `render/framing.py: decide(asset, slot)` (spec 07).
- `scores` from the cost breakdown; `reason`: up to 3 short phrases from the largest contributing terms, e.g. "top-ranked moment; matches 'beach at sunrise'; calm footage for a calm section".
- `notes`: fallbacks used, reuse mode, infeasible must-includes, truncated videos.
- Write `edl.json` to the run dir.

## 7. Baselines and ablations (`planning/baselines.py`)

Each returns an EDL with `system` set:
- `naive`: random candidates (seeded), fixed 2.0 s cuts ignoring beats (last shot trimmed to L), no director.
- `beat_random`: same slot grid as full, random candidates (respecting no-reuse).
- `greedy`: same grid; each slot takes the highest-importance unused candidate; no mood, pairwise, or director terms.
- Ablations of full via config overrides: `no_learned` (heuristic importance), `no_mood` (w_mood = w_val = 0), `no_pairwise` (all pairwise weights 0), `no_director` (w_dir = 0), `no_forced` (no forced section cuts).

## Acceptance tests

- `test_slots_invariants`: durations sum to L (+/- 1e-6); every cut on a beat; every shot within [min_shot_s, max_shot_s] except a final shot; forced boundaries present.
- `test_slots_pacing`: pacing +1 yields more slots than pacing -1 on the same song.
- `test_slots_scarcity`: with n_usable = 10 the grid has <= 9 slots or reuse_mode is set.
- `test_beam_constraints`: no candidate reused; no overlapping windows from one asset; cluster limit respected; excluded assets absent; must-include present.
- `test_beam_determinism`: same seed -> identical EDL; different seed -> different sequence.
- `test_beam_beats_greedy`: beam total cost <= greedy total cost under the same cost function.
- `test_mood_alignment`: synthetic candidates with known arousal and slots with known targets -> Spearman(chosen vr, ta) > 0.5.
- `test_boundary_contrast`: synthetic case -> mean vj at forced cuts > mean vj at ordinary cuts.
- `test_frame_quantization`: 40 shots at fps 30: sum of frame counts == round(L*30); no drift (each f_k == round(t_k*30)).
- `test_beam_speed`: K=40, M=120, B=128 under 2 s.
