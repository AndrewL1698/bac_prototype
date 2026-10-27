"""Framing decision per shot (16:9 output) and Ken Burns parameters for photos."""

from __future__ import annotations

import numpy as np

from montage.config import get_config
from montage.types import Asset, KenBurns

T = 16 / 9


def decide_framing(asset: Asset) -> str:
    cfg = get_config().render
    a = asset.width / max(asset.height, 1)
    if abs(a - T) / T < 0.02:
        return "native"
    if a < 1.1:
        return "fill_blur"
    if a < T:
        return "crop" if 1 - a / T <= cfg.max_crop_loss else "fill_blur"
    if a >= cfg.pano_aspect and asset.kind == "photo":
        return "pan"
    return "crop" if 1 - T / a <= cfg.max_crop_loss else "fill_blur"


def kenburns(asset: Asset, framing: str, intensity_rank: float, face_cx: float, face_cy: float,
             rng: np.random.Generator) -> KenBurns | None:
    """Photos only (not panoramas): slow zoom-in scaled by slot intensity, drift toward the face."""
    cfg = get_config().render
    if asset.kind != "photo":
        return None
    if framing == "pan":
        direction = 1.0 if rng.random() < 0.5 else -1.0
        return KenBurns(zoom_from=1.0, zoom_to=1.0, pan_x=direction, pan_y=0.0)
    zoom_to = 1.0 + cfg.kb_base + cfg.kb_gain * float(np.clip(intensity_rank, 0, 1))
    if face_cx >= 0:
        px = float(np.clip((face_cx - 0.5) * 0.2, -cfg.kb_max_pan, cfg.kb_max_pan))
        py = float(np.clip((face_cy - 0.5) * 0.2, -cfg.kb_max_pan, cfg.kb_max_pan))
    else:
        px, py = (float(v) for v in rng.uniform(-cfg.kb_max_pan, cfg.kb_max_pan, 2))
    return KenBurns(zoom_from=1.0, zoom_to=round(zoom_to, 4), pan_x=round(px, 4), pan_y=round(py, 4))
