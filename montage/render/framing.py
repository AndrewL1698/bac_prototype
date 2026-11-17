"""Framing decision per shot (16:9 output). Photos are still frames (no Ken Burns, decided 2026-11-15)."""

from __future__ import annotations

from montage.config import get_config
from montage.types import Asset

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
    return "crop" if 1 - T / a <= cfg.max_crop_loss else "fill_blur"

