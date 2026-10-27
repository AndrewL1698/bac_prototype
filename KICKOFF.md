# KICKOFF: starting and resuming Claude Code sessions

Repo: https://github.com/AndrewL1698/bac_prototype (public). Both collaborators follow "One-time setup"; only the first person does "Add the spec pack".

## One-time setup (each person, about 15 minutes)

1. **Repo access.** Andrew adds the other person as a collaborator (Settings -> Collaborators) so both can push. Clone locally:
   `git clone https://github.com/AndrewL1698/bac_prototype`
   Set your git name so logs attribute you: `git config user.name "Keshav"` (or "Andrew").
2. **Colab Secrets.** Open any Colab notebook, click the key icon, and add (enable notebook access for each):
   - `ANTHROPIC_API_KEY`: your own key (optional; without it the director uses rules)
   - `MONTAGE_USER`: `keshav` or `andrew`
   No GitHub token is needed: Colab only pulls, and the repo is public.
3. **Shared Drive folder.** Keshav owns `My Drive/montage_data` (already created, with `demo/`, `demo/songs/`, `eval/albums/`, `eval/songs/`) and shares it with Andrew as Editor. Andrew: open "Shared with me", right-click `montage_data` -> Organize -> Add shortcut -> My Drive. Both must see it at `My Drive/montage_data`.
4. **Claude Code + colab-mcp.** Use a **local** Claude Code session (Claude desktop app or `claude` CLI) opened on your clone of the repo. You need `uv` installed (`uvx` on PATH: `pip install uv` or `winget install astral-sh.uv`).
   - If the repo already has `.mcp.json` (someone committed it), just approve the `colab-mcp` server when Claude Code asks, then check with `/mcp`.
   - If not (first person only), run this once in a terminal in the repo folder, then commit and push the new `.mcp.json`:
     ```
     claude mcp add --transport stdio --scope project colab-mcp -- uvx git+https://github.com/googlecolab/colab-mcp
     ```
   - MCP timeouts: `.claude/settings.json` in the repo should contain the block below. If it is missing, ask Claude Code to create it and commit it.
     ```json
     { "env": { "MCP_TIMEOUT": "120000", "MCP_TOOL_TIMEOUT": "180000" } }
     ```
   The first start downloads colab-mcp and can take a minute. Restart the Claude Code session after adding the server so its tools load.
   A cloud Claude Code session works only if your computer is linked to it and the Claude GitHub App has access to this repo (Andrew would install it on his account); otherwise pushes are refused.
5. **Media.** Track A owner: demo album in `montage_data/demo/<album>/`, songs in `montage_data/demo/songs/`. Track B owner: 2 albums in `montage_data/eval/albums/`, 3 songs in `montage_data/eval/songs/`. Never put media in the repo.

During long jobs keep the Colab tab open and the laptop awake. When Claude Code says a prompt is waiting in the Colab tab (Drive mount, Secrets access, MCP connection), approve it there.

## Spec pack

Already committed on `main` (2026-10-26, Keshav). Tracks are proposed in `docs/PLAN.md` (Keshav: A, Andrew: B); swap there if you prefer.

## Start prompt (first session for each person)

```
You are building bac_prototype, an AI music video editor (Python package "montage"), in this repo.

Read CLAUDE.md fully, then docs/PLAN.md, the last 5 entries of docs/LOG.md, and docs/spec/schemas.md. These are binding. Each phase has its own spec in docs/spec/; read it right before implementing that phase.

I am <NAME> and I own Track <A or B> in docs/PLAN.md. Another person works in this repo in parallel, so:
- git fetch and pull main first.
- If .mcp.json or .claude/settings.json is missing, set them up first (KICKOFF.md step 4: run the `claude mcp add --scope project` command, create .claude/settings.json), commit and push them on main, then tell me to restart this session so the colab-mcp tools load, and stop until I do.
- Claim each phase in the "Current work" table of docs/PLAN.md and push that before starting. Never take a phase someone else claimed.
- Work on a branch per phase (phase-<N>-<slug>), merge to main only when the phase's acceptance tests pass, and follow the logging rules in CLAUDE.md section 4: append to docs/LOG.md at every phase end and session end, keep docs/PLAN.md and README.md current.

Heavy work (models, media, FFmpeg, datasets, training, the demo app) runs on my Colab A100 through colab-mcp (CLAUDE.md section 6). Never block a cell for more than ~25 seconds; use scripts/run_job.py for anything longer and poll it. Colab must run the branch you are on.

Track A: Day 1 goal is Phase 8 done: a Gradio share link where I upload media + a song + a brief and get a beat-synced video using fallbacks only. Day 2: 9A, then the demo parts of 11.
Track B: start dataset downloads and smoke tests as soon as the Phase 0 job runner is on main, then 9B, 10A, and the evaluation parts of 11. Check docs/PLAN.md dependencies before starting each.

If a spec is ambiguous, pick the simplest option consistent with CLAUDE.md, record it under Decisions in docs/PLAN.md, and keep going. Ask me only for things I must do myself (secrets, approvals in the Colab tab, media, taste calls, coordinating with the other person).

After each phase, tell me in 3-5 lines what works and what is next, then continue.
```

## Resume prompt (any later session)

```
Read CLAUDE.md, docs/PLAN.md, and the last 5 entries of docs/LOG.md. git fetch and rebase my branch if needed. Reconnect to Colab (CLAUDE.md section 6), check my running jobs with scripts/job_status.py, then continue my claimed phase, or claim the next unblocked phase on my track.
```

## End-of-day prompt

```
Wrap up: finish or checkpoint the current phase, make sure tests pass on my branch, push, append a docs/LOG.md entry (done, next, blockers, heads-up, results), update docs/PLAN.md "Current work", and tell me what the other person needs to know.
```
