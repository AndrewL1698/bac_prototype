# Phase 0: Setup, Colab ops, fixtures, background downloads

Goal: a skeleton repo that installs on Colab, a working background job system, synthetic fixtures, model smoke tests, and dataset downloads running in the background.

Repo: https://github.com/AndrewL1698/bac_prototype (public). Colab clones it to `/content/bac_prototype` without a token. The spec pack already provides `README.md`, `.gitignore`, `.gitattributes`, `configs/`, and `docs/`; extend them, do not replace them.

Split for two people: the **scaffold** (sections 1, 3, 4 and `test_repo_hygiene.py`) is done by whoever starts first and merged to `main` quickly; **smoke tests and downloads** (sections 5-6) can be done by the Track B owner as soon as the job runner is on `main`.

## 1. Repo skeleton

- `pyproject.toml` (setuptools, package `montage`, plus `training` and `scripts` importable), Python >= 3.10 (Colab's version).
- `requirements.txt`. Do **not** list `torch`/`torchvision`/`torchaudio` (Colab ships them; reinstalling breaks CUDA). List and pin after a successful install:
  `open_clip_torch, transformers, nnAudio, librosa, soundfile, scenedetect[opencv], opencv-python-headless, pillow, pillow-heif, numpy, scipy, scikit-learn, pydantic>=2, pyyaml, gradio, matplotlib, anthropic, mir_eval, tqdm, joblib, pytest`
  plus Beat This! (`pip install beat-this`; if that fails, `pip install git+https://github.com/CPJKU/beat_this`).
- `montage/config.py`: load `configs/default.yaml` into a nested pydantic model (`Config`), `get_config()` cached; allow overrides via `MONTAGE_CONFIG` (path to a second YAML merged on top) and dotted CLI overrides `--set planner.beam_width=64`.
- `montage/paths.py`: `DATA_ROOT` from env `MONTAGE_DATA_ROOT` (default `./data`), `SCRATCH` from `MONTAGE_SCRATCH` (default `/content/scratch` on Colab, `./data/scratch` locally), `USER` from `MONTAGE_USER` (default `anon`); helpers `run_dir(run_id)`, `cache_dir(kind)`, `jobs_dir()` -> `DATA_ROOT/jobs/<user>/`, `logs_dir()` -> `DATA_ROOT/logs/<user>/`, `new_run_id()` -> `<user>_<YYYYmmdd-HHMMSS>_<4 hex>`; create dirs on demand; `atomic_write(path, bytes|callable)` (temp file + `os.replace`) used for every cache/weight write on the shared Drive.
- `montage/types.py`: exactly `docs/spec/schemas.md`.
- `montage/log.py`: `stage(name)` context manager recording wall time into a per-run dict.
- `.gitignore` / `.gitattributes`: already provided; add entries if new artifact types appear.
- `tests/test_repo_hygiene.py`: as described in CLAUDE.md section 9. Uses `git ls-files`; skips cleanly if git is unavailable. Allow-list: `docs/img/*.png` under 500 KB.

## 2. Colab bootstrap

Every fresh runtime starts with one cell:

```
!git clone -q https://github.com/AndrewL1698/bac_prototype /content/bac_prototype 2>/dev/null; python /content/bac_prototype/scripts/colab_bootstrap.py --branch main
```

Before `colab_bootstrap.py` exists (very first Phase 0 session), run the equivalent by hand: mount Drive, clone, `pip install -e .` once the package exists.

`scripts/colab_bootstrap.py [--branch NAME]` must, idempotently:
1. Mount Drive if not mounted (skip if `/content/drive/MyDrive` exists). Check that `/content/drive/MyDrive/montage_data` exists; if not, print the shared-folder instructions from `KICKOFF.md` and stop.
2. Clone if missing, then `git fetch -q`, `git checkout -q NAME`, `git pull -q` (default `main`). The repo is public: no token. If `GITHUB_TOKEN` exists in Secrets (repo made private later), use it without ever printing it or the remote URL.
3. `pip install -q -r requirements.txt && pip install -q -e .`, only if the `requirements.txt` hash changed since the last install (stored in `/content/.montage_req_hash`).
4. Verify: `nvidia-smi --query-gpu=name,memory.total --format=csv`, `torch.cuda.is_available()`, FFmpeg version, and whether FFmpeg has the `zscale` and `tonemap` filters. If `zscale` is missing, download a static FFmpeg build into `/content/bin` and prepend it to PATH (record the choice in config `render.ffmpeg_bin`).
5. Read Secrets (`ANTHROPIC_API_KEY`, `MONTAGE_USER`) and write `/content/montage_env.sh` exporting `MONTAGE_DATA_ROOT=/content/drive/MyDrive/montage_data`, `MONTAGE_SCRATCH=/content/scratch`, `MONTAGE_USER`, PATH, and `ANTHROPIC_API_KEY` if present (file mode 600, never printed). `montage.paths.load_colab_env()` loads it for in-notebook use.
6. Print a one-line summary: branch and commit, user, GPU, CUDA ok, FFmpeg ok/zscale ok, API key present yes/no, shared Drive ok.

Job processes inherit env from `run_job.py`, which loads `/content/montage_env.sh` itself.

## 3. Background jobs

`scripts/run_job.py --name NAME [--force] -- CMD...`
- If a job with NAME is `running` for this user and its pid is alive, refuse (print status) unless `--force`.
- Warn (do not refuse) if another user has a `running` job with the same NAME (likely duplicate work on the shared Drive).
- Launch `setsid nohup CMD > DATA_ROOT/logs/<user>/NAME.log 2>&1 &` with env: `MONTAGE_JOB_NAME=NAME`, plus everything from `montage_env.sh`.
- Write `DATA_ROOT/jobs/<user>/NAME.json` as in CLAUDE.md section 7 (including branch and commit). A tiny watcher (the wrapper itself, double-forked) waits on the child and writes `done`/`failed` with the exit code.
- Return immediately, printing pid and log path.

`scripts/job_status.py [NAME] [--all-users]`: with NAME prints state, progress, message, elapsed, last 20 log lines. Without NAME lists this user's jobs in a table; `--all-users` lists everyone's (to see what the other person is running).

`montage/jobs.py`: `report(progress: float, message: str)` updates the status JSON at most every 5 s; `is_job()`.

## 4. Fixtures (`scripts/make_fixtures.py`)

Generate into `tests/fixtures/` with FFmpeg and numpy:
- `vid_motion_landscape.mp4`: 1920x1080, 30 fps, 12 s, `testsrc2` with a moving box (high motion).
- `vid_static_landscape.mp4`: 8 s, mostly static gradient with slight noise (low motion).
- `vid_portrait.mp4`: 1080x1920, 6 s.
- `vid_rotated.mov`: landscape-encoded with rotation metadata 90 (use `-metadata:s:v rotate=90` or display matrix).
- `vid_twoscenes.mp4`: 10 s, hard cut at 5 s between two different sources (scene detection test).
- `photo_*.jpg`: 8 gradient/pattern photos in different colors, one 4:3, one portrait 3:4, one panorama 3:1, one blurry (Gaussian blur), and two near-duplicates (same image, +2% brightness).
- `click_090.wav`, `click_120.wav`, `click_150.wav`: 70 s, 44.1 kHz, clicks on every beat, louder accented click every 4th beat (downbeat), plus quiet sine pad. Also `click_120_dynamic.wav` whose loudness steps up at 20 s and 40 s (for section/intensity tests).
- `song_sections.wav`: 90 s synthetic track with 3 clearly different sections (sparse/quiet, busy/loud, quiet) at 120 BPM.

Store ground truth (beat times, downbeat times, section boundaries) in `tests/fixtures/truth.json`.

## 5. Model smoke tests (`scripts/smoke_models.py`)

Each check prints PASS/FAIL and timing; failures do not abort the script. Run on Colab as a job.
1. **CLIP**: `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')`; encode 2 images + 2 texts; check the matching pair has higher cosine.
2. **LAION aesthetic head**: download `sac+logos+ava1-l14-linearMSE.pth` from the `christophschuhmann/improved-aesthetic-predictor` GitHub repo into `DATA_ROOT/weights/`; build the MLP (768->1024->128->64->16->1, dropouts as in the repo, no activations); score L2-normalized CLIP embeddings; expect values roughly 3-8.
3. **MERT**: `AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)` and `Wav2Vec2FeatureExtractor`, input 5 s at 24 kHz, `output_hidden_states=True`; expect 13 hidden states of shape (1, ~375, 768). If loading fails with the installed transformers, try pinning older transformers versions until it loads; record the working pin in requirements and PLAN.md Decisions.
4. **Beat This!**: run on `click_120.wav`; beats within 30 ms of truth for >= 95%; downbeats found.
5. **YuNet**: download `face_detection_yunet_2023mar.onnx` from opencv_zoo into `DATA_ROOT/weights/`; run `cv2.FaceDetectorYN` on a test image (no crash).
6. **Claude API** (only if `ANTHROPIC_API_KEY` set): one `client.messages.parse(...)` call with model from config `director.model` and a tiny pydantic `output_format`; print parsed result.
7. **FFmpeg**: encode 1 s of `testsrc` with config encode settings; `zscale`/`tonemap` availability.

## 6. Background downloads (start as soon as jobs work)

All write under `DATA_ROOT/datasets/`, are idempotent (skip if checksum/size matches), and report progress.

- `python -m training.emotion.download_deam`: from the DEAM page (`https://cvml.unige.ch/databases/DEAM/`) download the audio zip (~1.3 GB), annotations zip (~4.7 MB), and features zip (~600 MB). Parse the page for the actual links. If the host is down, try a Kaggle mirror only if the user provides Kaggle credentials; otherwise report and continue (emotion falls back to heuristic).
- `python -m training.importance.download`: Zenodo record 4884870 (SumMe + TVSum original videos + h5 annotation files, ~3.2 GB). List files via `https://zenodo.org/api/records/4884870`, download each, unzip.
- After DEAM finishes: queue `python -m training.emotion.extract_mert` (spec 09) as its own job.
- After SumMe/TVSum finish: queue `python -m training.importance.extract_clip` (spec 10).

## Acceptance tests

- `pytest -q` passes locally (non-gpu, non-media tests) and on Colab (all).
- `test_config_loads`: default config loads; dotted override works.
- `test_types_roundtrip`: every model in `types.py` serializes to JSON and back.
- `test_job_runner`: run a job `python -c "import time; [time.sleep(0.2) for _ in range(10)]"`; status goes `running` then `done`; a failing command ends `failed`; re-running a running job without `--force` is refused.
- `test_job_namespacing`: jobs started with `MONTAGE_USER=a` and `MONTAGE_USER=b` and the same name do not overwrite each other's status or logs.
- `test_atomic_write`: an interrupted write (simulated exception mid-write) leaves no partial file at the target path.
- `test_repo_hygiene`: passes on the repo; a temporary tracked `x.mp4` or a file containing `sk-ant-` makes it fail.
- `make_fixtures.py` produces every file listed and `truth.json`.
- `colab_bootstrap.py` summary line shows the branch, user, GPU, and "shared Drive ok".
- `smoke_models.py` passes CLIP, aesthetic, FFmpeg at minimum; MERT/Beat This!/YuNet/Claude results recorded in `docs/LOG.md` (and any version pins under Decisions in `docs/PLAN.md`).
- Download jobs are running (or done) and visible in `job_status.py --all-users`.
- `README.md` "What works today" updated; `docs/LOG.md` entry appended.
