# Director: brief and notes -> EditPlan -> planner directives

The user is the director. They type a brief ("Warm, nostalgic summer recap. Open slow on the beach at sunrise, build through the road trip, save the party for the drop, end on the sunset. Skip anything blurry.") and later give notes ("too fast in the middle", "more of the dog", "lose the selfie at 0:12").

The LLM only fills out an `EditPlan` (schemas.md). It never chooses frames. The optimizer executes the plan and keeps beat sync and constraints guaranteed. Sections 1-4 are Day 1 (Phase 4, rules only). Sections 5-8 are Day 2 (Phase 9A).

Privacy default: only text (brief, song summary, tag inventory, EDL summary) goes to the API. Images go only if the thumbnail toggle is on.

## 1. Context builders (`director/context.py`)

- `song_summary(profile_full, excerpt_candidates)`: from `music/profile.py: summarize`. For the first plan (before an excerpt is fixed) summarize the **auto-chosen excerpt** and also list song length so `excerpt_hint` can change it.
- `media_inventory(assets, candidates, features)`: one line per asset, max ~100 lines:
  `a012 video 0:24 2026-07-14 | beach, ocean, group of friends, swimming, sunset`
  Video tags = most frequent tags across its windows. Include kind, duration, date (if known).
- `edl_summary(edl)`: one line per shot: `0:00-0:03 a003 photo | beach, sunrise` (for notes).
- `allowed_queries_hint`: the tag vocabulary, so the LLM prefers phrases CLIP knows.

## 2. Rule-based fallback (`director/rules.py: plan_from_rules(brief) -> EditPlan`)

Deterministic, offline. Always available.
- `vibe` = brief (trimmed to 200 chars); `vibe_queries` = [brief] if brief else []; plus any tag-vocabulary words found in the brief as extra queries.
- Pacing keywords: slower (`slow, chill, calm, gentle, relaxed, mellow, cinematic`) -> -0.5; faster (`fast, hype, energetic, quick, intense, upbeat, punchy`) -> +0.5.
- Chronology: `chronological, in order, timeline, as it happened` -> true.
- Avoid: phrases after `skip|avoid|no|without|don't use` up to punctuation -> `avoid_queries`; `blurry` maps to "blurry photo".
- Opening/closing: "open on X", "start with X" -> opening_query X; "end on X", "finish with X" -> closing_query X.
- Length: "N seconds" / "Ns" -> target_length_s clamped to [45, 60].
- Everything else default: energy_bias 0, excerpt_hint auto, segments [], must_include [], exclude [], rationale "Rule-based plan (no LLM)."
- `source = "rules"`.

## 3. Directives (`director/apply.py: compile_plan(plan, profile, candidates, feats) -> Directives`)

`Directives` (internal dataclass) holds everything the slot builder and cost function need:

**Section mapping** for segments:
- `intro` -> section 0; `outro` -> last section; `peak` -> section containing `peak_time`; `build` -> the section(s) right before the peak section (none if peak is section 0); `section` -> `section_index` (ignore if out of range).
- Multiple segments on one section: merge their queries; last mood/pacing wins.

**Pacing per section**: effective p = segment pacing if nonzero else global `plan.pacing`, clamped to [-1, 1]. Slot builder multiplies base beats by `2^(-p)` before snapping (spec 06).

**Target arousal per slot** (rank space 0-1):
- segment mood `calm` 0.15, `neutral` 0.5, `energetic` 0.85, `match_music` or no segment -> slot's music-arousal rank.
- then `+ 0.25 * plan.energy_bias`, clamped to [0, 1].

**Query relevance** (precompute once per plan, per candidate):
- For a query set Q: `rel(i, Q) = max_q cos(e_i, text(q))`, then z-scored over candidates. Empty set -> zeros.
- `z_vibe`, `z_avoid`, `z_open`, `z_close`, and `z_seg[section]`.
- Director unary score for candidate i in slot j:
  `D(i,j) = w_vibe*z_vibe(i) + w_seg*z_seg[sec(j)](i) - w_avoid*relu(z_avoid(i) - avoid_margin) + [j==0]*w_open*z_open(i) + [j==last]*w_close*z_close(i)`
  Defaults: w_vibe 0.6, w_seg 1.2, w_avoid 1.5, avoid_margin 0.5, w_open 1.5, w_close 1.5.

**Hard sets**: `exclude` removes every candidate of those assets. `must_include` assets must appear (at least one candidate each) if feasible; infeasible ones produce an EDL note.

**Pins from the UI** (storyboard): pinned candidate ids act like must_include at candidate level; excluded candidate ids are removed.

**Chronology**: `plan.chronological` enables the chronology pairwise term.

**Length/excerpt**: `target_length_s` (0 -> config default) and `excerpt_hint`/`excerpt_start_s` pass to excerpt selection (spec 02).

## 4. Fallback flow (Day 1)

`director.make_plan(brief, ctx)`: if `director.use_llm` and an API key exists, use the LLM (section 5); on any failure or when disabled, use rules. Record `fallbacks` in `RunResult`.

## 5. LLM backend (`director/llm.py`), Day 2

Claude API, Python SDK `anthropic`. Model from config `director.model` (default `claude-haiku-5-5`, which supports structured outputs). Use **structured outputs**:

```python
import anthropic, json
client = anthropic.Anthropic()        # reads ANTHROPIC_API_KEY
schema = anthropic.transform_schema(EditPlanLLM)   # EditPlan minus `source`; all fields required
resp = client.messages.create(
    model=cfg.director.model, max_tokens=cfg.director.max_tokens,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_blocks}],
    output_config={"format": {"type": "json_schema", "schema": schema}},
    timeout=cfg.director.timeout_s,
)
```
- If `resp.stop_reason` is `refusal` or `max_tokens`, or parsing fails: retry once, then fall back to rules.
- Parse: `data = json.loads(text)`; lowercase all enum-valued fields (`excerpt_hint`, `anchor`, `mood`) before `EditPlan.model_validate` (structured outputs do not guarantee enum capitalization).
- Validate and repair: drop unknown asset ids, clamp pacing/energy_bias to [-1, 1], clamp target_length_s to 0 or [45, 60], cap each query list at `director.max_queries` (6) and each query at 8 words, drop empty strings. Set `source = "llm"`.
- Keep the schema simple: every field required (no optional fields, no unions) to stay well inside structured-output limits.

**System prompt** (store in `director/prompts.py`, keep under ~600 words). It must say:
- You are the assistant editor for a short music montage. The user is the director. You translate their brief into an edit plan; a separate optimizer picks and times the shots.
- Queries are searched with CLIP against the user's photos/videos. Write **concrete, visual** phrases of 2-6 words describing what a camera would see ("golden hour beach", "friends dancing at night"), not feelings or abstractions. Prefer words from the tag vocabulary and the media inventory.
- Map structure words to the song using the song summary: "the drop", "the chorus", "the climax" -> `peak`; "the beginning", "open" -> `intro`; "the end" -> `outro`; "build up" -> `build`; or a specific section index.
- Use `mood` = `match_music` unless the director asks for a feeling in that part.
- Leave `pacing` and `energy_bias` at 0 unless the director asks for speed/energy changes.
- Use `must_include`/`exclude` only with asset ids from the inventory, and only when the director clearly refers to specific items.
- Put things to skip in `avoid_queries` ("blurry photo", "screenshot").
- `rationale`: 1-2 short sentences in plain language explaining the plan. Do not include step-by-step reasoning.
- Never invent assets or content not suggested by the inventory.

**User content blocks**: brief; song summary; media inventory; tag vocabulary; (optional) contact sheet image (section 8).

## 6. Notes and recut (`director/notes.py`), Day 2

`revise_plan(prev_plan, edl, note) -> EditPlan`
- LLM call with: previous plan JSON, EDL summary (time-coded shots), song summary, inventory, the note. Instruction: return the full updated plan, changing only what the note asks; resolve time references ("at 0:12") via the EDL summary; "more of X" adds X to the relevant queries with emphasis; "lose/remove the shot at T" adds that asset to `exclude`; "too fast in the middle" lowers pacing for the middle section(s).
- Rules fallback: `faster`/`slower` +/-0.4 to global pacing (or to the section named by beginning/middle/end), `more X` appends X to vibe_queries, `less X`/`no X`/`lose X` appends to avoid_queries, `at m:ss` removal excludes the asset of the shot covering that time.
- Notes history is kept per session; each recut stores `plan.json` in the new run dir.

## 7. Editor's note (`director/editor_note.py`), Day 2

After planning, one plain-text call (no structured output, `max_tokens` 200): given plan + EDL summary + song summary, write two sentences explaining the cut in plain language with timestamps ("Opened on the sunrise beach shots, held the party clips for the drop at 0:32."). Fallback template: "Cut {n} shots to {tempo} BPM; strongest moments placed in the {peak section label} section at {peak_time}."

## 8. Thumbnail contact sheet (optional toggle)

`director.send_thumbnails` (default false) or the UI toggle. Build one JPEG grid of candidate thumbnails (one per asset, the most important window for videos), 128 px cells, asset id printed in each cell, max 100 cells, under ~1.5 MB. Add it as an image content block before the text and tell the model the ids are printed on each tile. Never send full-resolution media.

## Acceptance tests

- `test_rules_parse`: "chill, chronological, skip anything blurry, end on the sunset, 45 seconds" gives pacing -0.5, chronological true, avoid ["blurry photo"], closing_query "the sunset", target 45.
- `test_anchor_mapping`: on a profile with 4 sections and the peak in section 2: intro->0, build->1, peak->2, outro->3.
- `test_target_arousal`: calm segment gives 0.15 (+0.25 with energy_bias 1 -> 0.4).
- `test_director_scores_shape`: D has shape (N candidates, K slots) and exclude removes all windows of the asset.
- `test_llm_fallback_paths` (mocked client): timeout, refusal, max_tokens, bad JSON, wrong-case enums all produce a valid plan (rules for failures, lowercase-normalized for casing).
- `test_llm_live` (marked `api`, needs key): 3 sample briefs produce valid plans; every asset id exists; every query <= 8 words.
- `test_notes_rules`: "lose the shot at 0:12" excludes the asset covering 0:12 in a fixture EDL.
