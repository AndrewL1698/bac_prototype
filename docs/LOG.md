# LOG: progress history

Append-only. Newest entries at the bottom. Never edit or delete someone else's entry. This file uses git's `union` merge driver (`.gitattributes`), so two people appending at the same time never get a merge conflict locally; if GitHub's web UI reports a conflict on a PR, keep both entries.

Write an entry when you finish a phase, hit a blocker, make a decision that affects the other person, finish a training/eval run, or end a session.

Format:
```
## YYYY-MM-DD HH:MM ET | <person> | phase <N> | branch <name>
- Done: what now works (be concrete: commands, tests passing)
- Next: the next step you or someone else should take
- Blockers: what is stuck and what is needed (or none)
- Heads-up: changes to shared contracts, config keys, data layout, Drive folders (omit if none)
- Results: key numbers from runs (omit if none)
```

---

## 2026-10-26 21:15 ET | Keshav | setup | branch main
- Done: Committed the spec pack: README (how to use the repo), CLAUDE.md (architecture, collaboration and logging rules, Colab procedure), KICKOFF.md (setup + start/resume prompts), docs/PLAN.md, docs/LOG.md, docs/RESULTS.md, docs/spec/ (phases 00-11 + schemas), configs/default.yaml, configs/tags.txt, .gitignore, .gitattributes. No code yet.
- Done: Created the shared Drive folder `montage_data` in Keshav's My Drive with `demo/`, `demo/songs/`, `eval/albums/`, `eval/songs/`.
- Next: Keshav: add Colab Secrets (ANTHROPIC_API_KEY, MONTAGE_USER=keshav), put the demo album and songs in Drive, register colab-mcp at project scope (KICKOFF.md step 4) and commit .mcp.json + .claude/settings.json, then open a local Claude Code session on the clone and paste the start prompt (Track A, Phase 0). Andrew: confirm Track B in docs/PLAN.md, accept the Drive share and add a shortcut, add Colab Secrets (MONTAGE_USER=andrew), install uv, read README and KICKOFF.md.
- Blockers: none.
- Heads-up: tracks proposed as Keshav = A (pipeline, app, demo), Andrew = B (data, training, eval). Swap in docs/PLAN.md if you prefer.

## 2026-10-26 22:30 ET | Keshav | phase 0 | branch phase-0-setup
- Done: Committed `.mcp.json` (colab-mcp, project scope) and `.claude/settings.json` (MCP timeouts) on main. On `phase-0-setup`: pyproject, requirements, `montage/{config,paths,types,log,jobs,cache,weights}.py`, `scripts/{run_job,job_status,colab_bootstrap,make_fixtures,smoke_models}.py`, CLIP + aesthetic loaders, tests (config, types roundtrip, atomic write, job runner, job namespacing, repo hygiene). `pytest -q` green locally (15 passed).
- Next: verify on Colab (bootstrap, fixtures, smoke_models), merge phase 0 to main, then phase 1.
- Blockers: colab-mcp connects, but its notebook-cell tools did not appear in this Claude Code session; the project-scope server shows "pending approval". Keshav to approve it / restart the session.
- Heads-up: Colab bootstrap now runs with `%run` (Drive mount and Secrets only work in the notebook kernel): `!git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null` then `%run /content/bac_prototype/scripts/colab_bootstrap.py --branch <branch>`. Requirements install runs as background job `pip_install` when requirements.txt changes. Andrew: the job runner is on `phase-0-setup` until merged; downloads can start from that branch.

## 2026-10-26 23:20 ET | Keshav | phase 0 | branch phase-0-setup (merged)
- Done: Phase 0 merged to main. On Colab A100 (Python 3.13, torch 2.11 cu130, FFmpeg 6.1.1 with zscale/tonemap): bootstrap summary OK (branch, user keshav, GPU, CUDA, API key, shared Drive ok); `make_fixtures.py` writes all 21 fixtures + truth.json; `pytest -q` 16 passed; smoke: CLIP PASS, aesthetic PASS, Beat This! PASS (99% of beats within 30 ms on click_120), YuNet PASS, Claude API PASS (claude-haiku-5-5 structured output), FFmpeg PASS.
- Next: Keshav: phase 1 (ingest + candidates). Andrew: dataset downloads (DEAM, SumMe/TVSum) can start now; the job runner is on main.
- Blockers: MERT smoke FAIL under transformers 5.20 ("object of type 'NoneType' has no len()"); Track B to find a working pin (see PLAN Decisions). colab-mcp's notebook tools do not load in the Claude desktop app; Keshav's session drives Colab through a small local bridge built on the colab-mcp package instead.
- Heads-up: bootstrap is `%run scripts/colab_bootstrap.py --branch <b>` (see README). Colab Secrets cannot be read by remotely-run cells; run the bootstrap cell once from the Colab UI to save `ANTHROPIC_API_KEY`/`MONTAGE_USER` into `/content/montage_env.sh`. CLIP loads with QuickGELU (`force_quick_gelu=True`).
- Results: smoke {clip PASS (sims 0.27/0.31), aesthetic PASS (4.9/4.5), mert FAIL, beat_this PASS, yunet PASS, claude PASS, ffmpeg PASS}.
