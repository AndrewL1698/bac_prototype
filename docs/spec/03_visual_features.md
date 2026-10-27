# Phase 3: Visual features

Goal: give every candidate a numeric profile: CLIP embedding, aesthetic score, mood, tags, image quality, motion, faces, duplicate cluster.

## 0. Caching design

Compute and cache **per-frame** features per asset (keyed by asset hash + `features.version`):
`cache/features/<version>/<hash>.npz` with arrays aligned to that asset's analysis frames (photos: 1 row). Candidate-level features are aggregated on load (cheap), so changing window settings never re-runs models. Batch-relative values (z-scores, clusters) are computed after loading, never cached.

`features/extract.py: extract(assets, candidates) -> (FeatureTable, arrays)` loads cached assets, computes missing ones in batches (all frames of all missing assets go through CLIP together), then aggregates.

## 1. CLIP (`features/clip_encoder.py`)

- `open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")`, eval, fp16 on CUDA, batch `features.clip_batch` (128).
- Per frame: image embedding, L2-normalized, float32 at rest.
- Text encoding helper with an LRU cache (`encode_text(list[str]) -> (n, 768)` normalized).
- Candidate embedding: mean of its frames' embeddings, re-normalized (photos: the single embedding).

## 2. Aesthetic (`features/aesthetic.py`)

LAION aesthetic predictor V2 MLP (weights from Phase 0) applied to each **frame** embedding. Candidate value: mean over frames. Raw scale is roughly 1-10.

## 3. Image quality (`features/quality.py`)

Per frame, on the 448 px analysis image:
- `sharpness = log1p(var(Laplacian(gray)))`
- `brightness = mean(luma) / 255`
- `contrast = std(luma) / 255`
- `exposure_clip = fraction of pixels with luma < 5 or > 250`
- `colorfulness` = Hasler and Süsstrunk (2003): rg = R - G, yb = 0.5(R + G) - B; `sqrt(std_rg^2 + std_yb^2) + 0.3 * sqrt(mean_rg^2 + mean_yb^2)`, divided by 100.
Candidate value: median over frames.

## 4. Motion (`features/motion.py`), videos only

For each motion pair (t, t + 1/15 s) at 320 px: Farneback dense flow (`cv2.calcOpticalFlowFarneback`, standard params).
- `mag` = mean flow magnitude / frame diagonal (resolution independent).
- global motion vector = median (dx, dy).
Candidate: `motion` = mean `mag` over its pairs; `shake` = std of the global motion vectors' frame-to-frame differences (high-frequency jitter), normalized by diagonal. Photos: `motion = 0`, `shake = 0`.

## 5. Faces (`features/faces.py`)

YuNet (`cv2.FaceDetectorYN`, score threshold `features.face_score` 0.7) on analysis frames.
- `face_count` = max over frames; `face_area` = max fraction of frame area covered by the largest face.
- `face_cx, face_cy` = center of the union box of faces in the frame nearest the window center (normalized 0-1; -1 if none). Used by framing (crop placement).

## 6. Mood (`features/mood.py`)

Prompt ensembles (in `configs/default.yaml` under `mood.prompts`), each phrase wrapped in templates `"a photo of {}"` and `"a video still of {}"`:
- arousal high: "an energetic, exciting moment", "people dancing at a party", "fast action and movement", "a cheering crowd", "an adrenaline-filled moment"
- arousal low: "a calm, peaceful scene", "a quiet, still moment", "a serene landscape", "a relaxing slow afternoon", "a soft, gentle scene"
- valence positive: "a happy, joyful moment", "friends laughing together", "a bright cheerful day", "a celebration", "a warm, loving moment"
- valence negative: "a sad, melancholic scene", "a gloomy, dark day", "a lonely moment", "a tense, somber scene", "a cold, empty place"

Pole embedding = normalized mean of its phrase/template text embeddings.
Raw scores per candidate: `arousal_raw = 100 * (cos(e, high) - cos(e, low))`, same for valence. Cached raw.

At batch time (`mood.combine`), z-score within the current upload set, then:
- `visual_arousal = 0.5*z(arousal_raw) + 0.3*z_motion + 0.2*z(colorfulness)` where `z_motion` is z(motion) over video candidates and the constant `mood.photo_motion_z` (-0.5) for photos.
- `visual_valence = 0.6*z(valence_raw) + 0.2*z(brightness) + 0.2*z(colorfulness)`
Weights in config.

## 7. Tags (`features/tags.py`)

Vocabulary: `configs/tags.txt` (provided, ~110 concrete concepts, one per line; extend if needed). It covers:
- scenes: beach, ocean, lake, river, mountain, forest, desert, snow, city street, skyline, park, garden, road, highway, airport, airplane window, train, boat, pool, stadium, concert, nightclub, restaurant, cafe, kitchen, living room, bedroom, office, classroom, museum, church, market, campfire, sunset, sunrise, night, fireworks, rain, clouds, stars
- subjects: person, group of friends, couple, family, child, baby, crowd, selfie, portrait, dog, cat, horse, bird, food, dessert, drinks, cake, car, bicycle, motorcycle, building, bridge, landmark, flowers, trees
- activities: dancing, singing, swimming, surfing, skiing, hiking, running, playing sports, soccer, basketball, eating, cooking, laughing, hugging, jumping, driving, walking, shopping, graduation, wedding, birthday party, road trip, camping, gaming, studying, working out
- qualities: screenshot, text document, blurry photo, dark photo, meme, receipt (useful for avoid queries)
Text embedding per tag with template "a photo of {tag}". For each candidate keep the top `features.tags_top_k` (5) with cosine >= `features.tag_min_cos` (0.2).

## 8. Duplicates (`features/dedup.py`)

Agglomerative clustering (average linkage, cosine distance) over candidate embeddings with distance threshold `1 - dedup.cos_threshold` (0.92). Rule: **windows from the same video are never merged with each other** (they overlap by design; the planner handles same-source limits). Implement by clustering all, then splitting any cluster into per-asset sub-clusters only when it contains windows of a single video. Output `dup_cluster` int per candidate.

## Acceptance tests

- `test_clip_text_match` (gpu): an image of a red square matches "a red square" over "a blue circle" (generate with PIL).
- `test_feature_cache` : second extraction on the same assets runs zero CLIP forwards.
- `test_quality_blur`: the blurry fixture photo has the lowest sharpness among fixture photos.
- `test_motion_order`: `vid_motion_landscape` candidates have higher mean motion than `vid_static_landscape`.
- `test_dedup_neardup`: the two near-duplicate photos share a cluster; distinct photos do not; adjacent windows of one video are in different clusters.
- `test_mood_batch_z`: `visual_arousal`/`visual_valence` have mean ~0 and std ~1 over a batch.
- `test_tags_shape`: every candidate has <= 5 tags, all from the vocabulary.
- Timing (gpu): 60 photos + 20 videos x 30 s, cold cache, under 90 s total for this phase on the A100.
