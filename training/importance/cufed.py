"""CUFED / ML-CUFED photo albums for the importance ranker (phase 15, spec 15 sections 1-3).

    python scripts/run_job.py --name cufed_dl -- python -m training.importance.cufed
    python -m training.importance.cufed --inspect          # report what is on disk; downloads nothing
    python -m training.importance.cufed --set importance.cufed.flickr.enabled=true   # also fill from Flickr

1. Find or fetch the dataset, trying the sources in `importance.cufed.sources` (manual -> official -> hf) on
   every run. Each remote source keeps its own folder (raw/official, raw/hf) and is reused once on disk; "manual"
   is any other copy under raw/. The dataset root is the folder that holds image_importance.json.
2. Optionally fetch labeled photos missing on disk from their Flickr URLs (off by default).
3. Write the label mapping DATA_ROOT/datasets/importance/cufed/mapping.json: one group per album, one item
   per labeled photo, y = min-max of the score within the album's photos on disk.

Layout (spec 15 section 1): image_importance.json {album: [[flickr_url, "album/photo", score], ...]},
event_type.json {album: [event, ...]}, images/<album>/<photo>.jpg. Idempotent; data stays on the Drive.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from montage import cache, config, jobs, paths
from montage.config import CufedCfg, get_config
from montage.ingest.assets import PHOTO_EXT
from montage.log import get_logger
from training.common import datasets_dir, download

log = get_logger("cufed")
IMPORTANCE_FILE = "image_importance.json"
EVENTS_FILE = "event_type.json"
ARCHIVE_EXT = (".zip", ".tar", ".tgz", ".tar.gz")
DOWNLOADED = "_downloaded.json"
FLICKR_FAILED = "_flickr_failed.json"
USER_AGENT = "bac_prototype-cufed/1.0 (research dataset fetch)"
NOT_CUFED_HINT = ("no image_importance.json found. This is not the event-curation CUFED: the 'CUFED'/'CUFED5' "
                  "sets used by super-resolution papers (TTSR, SRNTT, C2-Matching) are image patches without "
                  "importance labels. See docs/spec/15_cufed.md section 1.")


@dataclass(frozen=True)
class LabeledPhoto:
    album: str
    photo: str          # photo id = file stem, e.g. "8725968285"
    score: float        # raw CUFED importance (mean rating, about -2 to 2)
    url: str | None     # original Flickr URL


# ---------- locations ----------

def cufed_dir() -> Path:
    d = datasets_dir("importance") / "cufed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_dir() -> Path:
    d = cufed_dir() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mapping_path() -> Path:
    return cufed_dir() / "mapping.json"


def _skip_name(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return "__MACOSX" in parts or any(p.startswith("._") for p in parts)


# ---------- dataset root and labels ----------

def find_root(base: Path, exclude: tuple[str, ...] = ()) -> Path | None:
    """The folder holding image_importance.json under `base` (top-level folders in `exclude` are not searched;
    images/ folders are never walked, they hold the photos). With several, the one with the most album folders
    wins (ties: the shallowest)."""
    if not base.exists():
        return None
    hits = []
    for dirpath, dirnames, filenames in os.walk(base):
        d = Path(dirpath)
        dirnames[:] = sorted(n for n in dirnames if not n.startswith(".") and n not in ("__MACOSX", "images")
                             and not (d == base and n in exclude))
        if IMPORTANCE_FILE in filenames:
            hits.append(d)
    if not hits:
        return None

    def n_album_dirs(root: Path) -> int:
        imgs = root / "images"
        return sum(1 for d in imgs.iterdir() if d.is_dir()) if imgs.is_dir() else 0

    return sorted(hits, key=lambda r: (-n_album_dirs(r), len(r.parts), str(r)))[0]


def parse_importance(data) -> dict[str, list[LabeledPhoto]]:
    """image_importance.json -> {album: [LabeledPhoto]}. Entries are [url, "album/photo", score] (verified
    layout); dicts with url/key/score are also accepted. Duplicate photo ids in one album keep the first."""
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{IMPORTANCE_FILE}: expected a non-empty JSON object of albums")
    out: dict[str, list[LabeledPhoto]] = {}
    for album, entries in data.items():
        if not isinstance(entries, list):
            raise ValueError(f"{IMPORTANCE_FILE}: album {album!r} is not a list")
        seen: set[str] = set()
        photos = []
        for e in entries:
            if isinstance(e, (list, tuple)) and len(e) >= 3:
                url, key, score = e[0], e[1], e[2]
            elif isinstance(e, dict):
                url, key, score = e.get("url"), e.get("key") or e.get("image") or e.get("photo"), e.get("score")
            else:
                raise ValueError(f"{IMPORTANCE_FILE}: unexpected entry in album {album!r}: {str(e)[:80]}")
            if key is None or score is None:
                raise ValueError(f"{IMPORTANCE_FILE}: entry without a photo key or score in album {album!r}")
            photo = str(key).replace("\\", "/").split("/")[-1]
            if Path(photo).suffix.lower() in PHOTO_EXT:
                photo = Path(photo).stem
            try:
                s = float(score)
            except (TypeError, ValueError):
                raise ValueError(f"{IMPORTANCE_FILE}: non-numeric score {score!r} in album {album!r}") from None
            if photo in seen:
                continue
            seen.add(photo)
            photos.append(LabeledPhoto(str(album), photo, s, str(url) if url else None))
        out[str(album)] = photos
    return out


def read_labels(root: Path) -> dict[str, list[LabeledPhoto]]:
    return parse_importance(json.loads((root / IMPORTANCE_FILE).read_text(encoding="utf-8")))


def read_events(root: Path) -> dict[str, list[str]]:
    p = root / EVENTS_FILE
    if not p.exists():
        log.warning("%s missing next to %s; every album gets event 'unknown'", EVENTS_FILE, IMPORTANCE_FILE)
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): ([v] if isinstance(v, str) else [str(x) for x in v]) for k, v in data.items()}


def index_album_dirs(root: Path, albums: set[str], max_depth: int = 3) -> dict[str, Path]:
    """{album id: its folder}. Usually root/images/<album>, but archives can nest one level more
    (e.g. images/<event>/<album>), so any folder named like a labeled album within max_depth counts."""
    out: dict[str, Path] = {}
    base = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        dirnames[:] = sorted(n for n in dirnames if not n.startswith(".") and n != "__MACOSX")
        if d != root and d.name in albums and d.name not in out:
            out[d.name] = d
            dirnames[:] = []
        elif len(d.parts) - base >= max_depth:
            dirnames[:] = []
    return out


def album_files(folder: Path | None) -> dict[str, Path]:
    """{photo id (stem): file} for the photo files ingest accepts."""
    if folder is None or not folder.is_dir():
        return {}
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in PHOTO_EXT:
            out.setdefault(p.stem, p)
    return out


def dataset_stats(root: Path) -> dict:
    """Validates the root (raises ValueError) and counts what is there."""
    labels = read_labels(root)
    index = index_album_dirs(root, set(labels))
    n_files = sum(len(album_files(d)) for d in index.values())
    return {"albums_labeled": len(labels), "photos_labeled": sum(len(v) for v in labels.values()),
            "albums_with_folder": len(index), "photo_files": n_files,
            "has_event_types": (root / EVENTS_FILE).exists()}


# ---------- archives ----------

def _archive_kind(path: Path) -> str | None:
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:4] == b"PK\x03\x04":
        return "zip"
    return "tar" if tarfile.is_tarfile(path) else None


def safe_extract(archive: Path, dest: Path) -> int:
    """Extract a zip/tar into dest. Members that would land outside dest (absolute paths, '..') and links
    are refused; macOS metadata is skipped. A marker makes reruns skip the archive. Returns files written."""
    marker = dest / f"_extracted_{archive.name}.json"
    done = cache.read_json(marker)
    if done:
        return int(done.get("files", 0))
    kind = _archive_kind(archive)
    if kind is None:
        raise ValueError(f"{archive.name} is neither a zip nor a tar archive")
    dest.mkdir(parents=True, exist_ok=True)
    dest_r = dest.resolve()
    written = refused = 0

    def target(name: str) -> Path | None:
        t = (dest / name.replace("\\", "/").lstrip("/")).resolve()
        if Path(name.replace("\\", "/")).is_absolute() or (t != dest_r and dest_r not in t.parents):
            return None
        return t

    def put(name: str, size: int, opener) -> None:
        nonlocal written, refused
        if _skip_name(name):
            return
        t = target(name)
        if t is None:
            refused += 1
            log.warning("%s: refused member outside the target folder: %s", archive.name, name)
            return
        if not (t.exists() and t.stat().st_size == size):
            with opener() as src:
                paths.atomic_write(t, src.read())
        written += 1
        if written % 500 == 0:
            jobs.report(0.65, f"extracting {archive.name}: {written} files")

    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            for i in zf.infolist():
                is_link = ((i.external_attr >> 16) & 0o170000) == 0o120000
                if not i.is_dir() and not is_link:
                    put(i.filename, i.file_size, lambda i=i: zf.open(i))
    else:
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if m.isfile():  # skips dirs, symlinks, hard links, devices
                    put(m.name, m.size, lambda m=m: tf.extractfile(m))
    cache.write_json(marker, {"archive": archive.name, "files": written, "refused": refused})
    return written


def extract_all(folder: Path, max_rounds: int = 3) -> int:
    """Extract every archive under folder in place (next to the archive), including archives that come out
    of archives, up to max_rounds levels."""
    total = 0
    for _ in range(max_rounds):
        todo = [a for a in sorted(folder.rglob("*")) if a.is_file() and a.name.lower().endswith(ARCHIVE_EXT)
                and not _skip_name(str(a.relative_to(folder)))
                and not (a.parent / f"_extracted_{a.name}.json").exists()]
        if not todo:
            break
        for a in todo:
            total += safe_extract(a, a.parent)
    return total


# ---------- sources ----------
# Each remote source downloads into its own folder under raw/ ("manual" = anything else under raw/), so a later
# official link or a manual copy is never shadowed by an earlier mirror download: every run walks the sources in
# priority order and reuses a source's folder when it already holds a valid root.

SOURCE_DIRS = {"official": "official", "hf": "hf"}
SOURCE_INFO = "_source.json"


def local_root(src: str) -> Path | None:
    if src == "manual":
        return find_root(raw_dir(), exclude=tuple(SOURCE_DIRS.values()))
    return find_root(raw_dir() / SOURCE_DIRS[src])


def fetch_official(url: str | None) -> tuple[Path | None, dict]:
    if not url:
        log.info("official: importance.cufed.official_url is not set; skipped")
        return None, {}
    name = Path(urllib.parse.urlparse(url).path).name or "archive"
    try:
        arc = download(url, paths.scratch_dir("downloads") / f"cufed_official_{name}", 0.0, 0.6, "CUFED (official)")
    except Exception as e:  # dead link, no network: try the next source
        log.warning("official: download failed (%s: %s); trying the next source", type(e).__name__, e)
        return None, {}
    with open(arc, "rb") as f:
        head = f.read(512).lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        arc.unlink(missing_ok=True)
        raise SystemExit("official_url returned a web page, not an archive (e.g. a Google Drive warning page). "
                         f"Download it in a browser and copy the extracted folder into {raw_dir()} "
                         "(the 'manual' source picks it up).")
    dest = raw_dir() / SOURCE_DIRS["official"]
    safe_extract(arc, dest)
    extract_all(dest)
    info = {"url": url, "downloaded": datetime.now().isoformat(timespec="seconds")}
    cache.write_json(dest / SOURCE_INFO, info)
    return find_root(dest), info


def fetch_hf(repo: str, revision: str) -> tuple[Path | None, dict]:
    try:
        from huggingface_hub import HfApi, snapshot_download
        sha = HfApi().dataset_info(repo, revision=revision).sha
        dest = raw_dir() / SOURCE_DIRS["hf"]
        jobs.report(0.05, f"downloading {repo}@{(sha or revision)[:8]}")
        snapshot_download(repo_id=repo, repo_type="dataset", revision=sha or revision, local_dir=str(dest))
    except Exception as e:  # missing repo, gated repo without HF_TOKEN, network: try the next source
        log.warning("hf: could not download %s@%s (%s: %s). If the repo needs a login, accept its terms on "
                    "huggingface.co and set the HF_TOKEN Colab Secret.", repo, revision, type(e).__name__, e)
        return None, {}
    jobs.report(0.6, "extracting archives")
    extract_all(dest)
    info = {"repo": repo, "revision": revision, "sha": sha, "downloaded": datetime.now().isoformat(timespec="seconds")}
    cache.write_json(dest / SOURCE_INFO, info)
    return find_root(dest), info


def ensure_dataset(cfg: CufedCfg, inspect_only: bool = False) -> tuple[Path | None, dict]:
    """(dataset root, info): the first source in cfg.sources that has (or downloads) a valid root. A source
    already on disk is reused without any network call. inspect_only never downloads and never writes
    raw/_downloaded.json (the record of which root the mapping used)."""
    for src in cfg.sources:
        root, extra = local_root(src), {}
        if root is not None and src in SOURCE_DIRS:
            extra = cache.read_json(raw_dir() / SOURCE_DIRS[src] / SOURCE_INFO) or {}
        elif root is None and src in SOURCE_DIRS and not inspect_only:
            root, extra = (fetch_official(cfg.official_url) if src == "official"
                           else fetch_hf(cfg.hf_repo, cfg.hf_revision))
        if root is None:
            has_files = any(p for p in raw_dir().iterdir() if not p.name.startswith("_"))
            if src == "manual" and has_files:
                log.warning("manual: files under %s but %s", raw_dir(), NOT_CUFED_HINT)
            else:
                log.info("%s: nothing found", src)
            continue
        try:
            stats = dataset_stats(root)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("%s: %s is not a usable CUFED root: %s", src, root, e)
            continue
        info = {"source": src, "root": str(root), **extra, **stats,
                "found": datetime.now().isoformat(timespec="seconds")}
        if not inspect_only:
            cache.write_json(raw_dir() / DOWNLOADED, info)
        return root, info
    return None, {}


# ---------- optional Flickr fill ----------

def fetch_image(url: str, target: Path, timeout: float) -> str | None:
    """Download one photo; None on success, else the reason. Only JPEGs that decode are kept: Flickr answers
    deleted photos with a 'photo unavailable' placeholder image, which must not become a training photo."""
    from PIL import Image
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            final = r.geturl() or url
            data = r.read()
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:200]
    if "unavailable" in final.lower():
        return "flickr placeholder (photo unavailable)"
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except Exception as e:
        return f"not an image: {type(e).__name__}"
    if fmt != "JPEG":
        return f"not a JPEG ({fmt}); likely a placeholder"
    paths.atomic_write(target, data)
    return None


def flickr_fill(root: Path, labels: dict[str, list[LabeledPhoto]], index: dict[str, Path], cfg: CufedCfg,
                retry_failed: bool = False) -> dict:
    """Fetch labeled photos that are missing on disk. Dead links go to raw/_flickr_failed.json and are skipped
    on later runs unless retry_failed."""
    fc = cfg.flickr
    failed_p = raw_dir() / FLICKR_FAILED
    failed: dict[str, str] = cache.read_json(failed_p) or {}
    todo: list[tuple[LabeledPhoto, Path]] = []
    for album, photos in sorted(labels.items()):
        folder = index.get(album) or root / "images" / album
        have = album_files(folder)
        for p in photos:
            key = f"{album}/{p.photo}"
            if p.photo in have or not p.url or (key in failed and not retry_failed):
                continue
            todo.append((p, folder / f"{p.photo}.jpg"))
    log.info("flickr: %d labeled photos to fetch (%d known dead links skipped)", len(todo),
             0 if retry_failed else len(failed))

    def one(item: tuple[LabeledPhoto, Path]) -> tuple[str, str | None]:
        p, target = item
        time.sleep(fc.delay_s)
        return f"{p.album}/{p.photo}", fetch_image(p.url, target, fc.timeout_s)

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, fc.workers)) as ex:
        for i, (key, err) in enumerate(ex.map(one, todo), 1):
            if err is None:
                ok += 1
                failed.pop(key, None)
            else:
                failed[key] = err
            if i % 100 == 0 or i == len(todo):
                jobs.report(0.7 + 0.25 * i / max(len(todo), 1), f"flickr {i}/{len(todo)} ({ok} ok)")
                cache.write_json(failed_p, failed)
    cache.write_json(failed_p, failed)
    return {"tried": len(todo), "fetched": ok, "failed": len(todo) - ok, "known_dead_total": len(failed)}


# ---------- label mapping ----------

def build_mapping(root: Path, labels: dict[str, list[LabeledPhoto]], events: dict[str, list[str]],
                  index: dict[str, Path], cfg: CufedCfg, out: Path | None = None) -> dict:
    """One group per album, one item per labeled photo on disk; y = min-max within those photos.
    Every labeled album ends up exactly once in `albums` or `dropped`; every kept photo maps to one file."""
    lo, hi = cfg.score_range
    albums: dict[str, dict] = {}
    dropped: dict[str, dict] = {}
    used: set[str] = set()
    out_of_range = 0
    for album in sorted(labels):
        photos = labels[album]
        bad = [p for p in photos if not math.isfinite(p.score)]
        if bad:
            raise SystemExit(f"non-finite score in album {album}: {bad[0]}")
        out_of_range += sum(not (lo <= p.score <= hi) for p in photos)
        files = album_files(index.get(album))
        present = [(p, files[p.photo]) for p in photos if p.photo in files]
        reason = ("no_images" if not files else "few_photos" if len(present) < cfg.min_photos else None)
        s = np.array([p.score for p, _ in present], float)
        if reason is None and s.max() - s.min() <= 0:
            reason = "tied_scores"
        if reason:
            dropped[album] = {"reason": reason, "n_labeled": len(photos), "n_present": len(present)}
            continue
        y = (s - s.min()) / (s.max() - s.min())
        rows = []
        for (p, f), yi in zip(present, y):
            key = str(f.resolve())
            if key in used:
                raise SystemExit(f"photo file mapped twice: {f}")
            used.add(key)
            rows.append({"photo": p.photo, "path": str(f), "score": p.score, "y": round(float(yi), 6)})
        albums[album] = {"events": events.get(album) or ["unknown"], "photos": rows,
                         "n_labeled": len(photos), "n_missing": len(photos) - len(present)}
    if set(albums) & set(dropped) or set(albums) | set(dropped) != set(labels):
        raise SystemExit("mapping does not cover every labeled album exactly once")
    if out_of_range:
        log.warning("%d scores outside score_range %s", out_of_range, cfg.score_range)
    summary = {
        "root": str(root), "albums_labeled": len(labels), "albums_kept": len(albums),
        "photos_kept": sum(len(a["photos"]) for a in albums.values()),
        "photos_missing_in_kept": sum(a["n_missing"] for a in albums.values()),
        "dropped_by_reason": dict(Counter(d["reason"] for d in dropped.values())),
        "albums_by_event": dict(Counter(a["events"][0] for a in albums.values()).most_common()),
        "multi_label_albums": sum(len(a["events"]) > 1 for a in albums.values()),
        "scores_out_of_range": out_of_range, "min_photos": cfg.min_photos,
    }
    mapping = {"albums": albums, "dropped": dropped, "summary": summary}
    cache.write_json(out or mapping_path(), mapping)
    return mapping


def load_mapping(path: Path | None = None) -> dict:
    m = cache.read_json(path or mapping_path())
    if not m:
        raise SystemExit(f"{path or mapping_path()} not found: run python -m training.importance.cufed first")
    return m


# ---------- entry point ----------

GUIDANCE = """No CUFED dataset found. Options (docs/spec/15_cufed.md section 2):
  1. manual:   download CUFED in a browser and copy the extracted folder (the one with image_importance.json)
               into {raw}
  2. official: copy the "Download CUFED" link from https://acsweb.ucsd.edu/~yuw176/event-curation.html and run
               with --set importance.cufed.official_url=<link>
  3. hf:       the Hugging Face mirror ({repo}); check the log above for why it failed (e.g. HF_TOKEN)."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CUFED download + label mapping (spec 15)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--inspect", action="store_true", help="report what is on disk; download nothing")
    ap.add_argument("--retry-failed", action="store_true", help="retry Flickr links that failed before")
    a = ap.parse_args(argv)
    paths.load_colab_env()
    if a.set:
        config.set_overrides(a.set)
    cfg = get_config().importance.cufed
    root, info = ensure_dataset(cfg, inspect_only=a.inspect)
    if root is None:
        print(GUIDANCE.format(raw=raw_dir(), repo=cfg.hf_repo), flush=True)
        return 2
    print("CUFED root:", json.dumps(info), flush=True)
    if a.inspect:
        return 0
    labels, events = read_labels(root), read_events(root)
    index = index_album_dirs(root, set(labels))
    if cfg.flickr.enabled:
        stats = flickr_fill(root, labels, index, cfg, retry_failed=a.retry_failed)
        print("Flickr fill:", json.dumps(stats), flush=True)
        index = index_album_dirs(root, set(labels))
    jobs.report(0.97, "label mapping")
    mapping = build_mapping(root, labels, events, index, cfg)
    print("CUFED mapping:", json.dumps(mapping["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
