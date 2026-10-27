"""Scene detection with PySceneDetect on the 360p proxy written during frame extraction."""

from __future__ import annotations

from montage import cache
from montage.config import get_config
from montage.ingest.frames import FrameSet
from montage.log import get_logger
from montage.types import Asset

log = get_logger("scenes")
SCENES_VERSION = 1


def detect_scenes(asset: Asset, frames: FrameSet, limit_s: float | None = None) -> list[tuple[float, float]]:
    """[(start, end)] in source seconds covering [0, duration or limit]. Cached by asset hash + settings."""
    cfg = get_config().candidates
    end_all = min(asset.duration or 0.0, limit_s or 1e9)
    key = f"{asset.hash}_{cfg.scene_detector}_{cfg.min_scene_s}_{round(end_all, 2)}_v{SCENES_VERSION}"
    cp = cache.cache_path("scenes", key, ".json")
    hit = cache.read_json(cp)
    if hit is not None:
        return [tuple(x) for x in hit]
    scenes: list[tuple[float, float]] = []
    try:
        from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video
        src = frames.proxy or asset.src_path
        video = open_video(src)
        fps = float(video.frame_rate or 15.0)
        min_len = max(1, int(round(cfg.min_scene_s * fps)))
        det = AdaptiveDetector(min_scene_len=min_len) if cfg.scene_detector == "adaptive" \
            else ContentDetector(min_scene_len=min_len)
        sm = SceneManager()
        sm.auto_downscale = True
        sm.add_detector(det)
        sm.detect_scenes(video, show_progress=False)
        for s, e in sm.get_scene_list(start_in_scene=True):
            scenes.append((round(s.get_seconds(), 3), round(e.get_seconds(), 3)))
    except Exception as e:  # noqa: BLE001  scene detection is best-effort
        log.warning("scene detection failed for %s (%s); using one scene", asset.filename, e)
        scenes = []
    if not scenes:
        scenes = [(0.0, round(end_all, 3))]
    else:
        scenes[0] = (0.0, scenes[0][1])
        scenes[-1] = (scenes[-1][0], max(scenes[-1][1], round(end_all, 3)))
        scenes = [(s, min(e, end_all)) for s, e in scenes if s < end_all]
    cache.write_json(cp, scenes)
    return scenes
