"""Phase 0 acceptance: make_fixtures.py produces every listed file and truth.json."""

from __future__ import annotations

import pytest

EXPECTED = [
    "vid_motion_landscape.mp4", "vid_static_landscape.mp4", "vid_portrait.mp4", "vid_rotated.mov",
    "vid_twoscenes.mp4", "vid_hdr.mp4", "ref_rotated_upright.mp4",
    "photo_01_sunset.jpg", "photo_02_checker.jpg", "photo_03_circles_4x3.jpg", "photo_04_stripes_portrait.jpg",
    "photo_05_pano.jpg", "photo_06_blurry.jpg", "photo_07_dupA.jpg", "photo_08_dupB.jpg",
    "click_090.wav", "click_120.wav", "click_150.wav", "click_120_dynamic.wav", "song_sections.wav",
    "truth.json",
]


@pytest.mark.media
def test_fixtures_complete(fixtures, truth):
    missing = [f for f in EXPECTED if not (fixtures / f).exists()]
    assert not missing, f"missing fixtures: {missing}"
    assert truth["click_120.wav"]["bpm"] == 120
    assert truth["song_sections.wav"]["boundaries"] == [30.0, 60.0]
    assert truth["vid_rotated.mov"]["probe_rotation"] is not None
