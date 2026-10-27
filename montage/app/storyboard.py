"""Storyboard: shot thumbnails with time codes (gallery items + one JPEG sheet)."""

from __future__ import annotations

from pathlib import Path

from montage.music.profile import mmss
from montage.types import EDL, Candidate


def gallery_items(edl: EDL, cands: dict[str, Candidate]) -> list[tuple[str, str]]:
    return [(cands[s.candidate_id].thumb_path, f"{mmss(s.start)}  {s.asset_id} {s.kind}")
            for s in edl.shots if s.candidate_id in cands]


def storyboard_jpg(edl: EDL, cands: dict[str, Candidate], out: str | Path, cols: int = 8, cell: int = 192) -> Path:
    from PIL import Image, ImageDraw
    items = gallery_items(edl, cands)
    rows = max(1, (len(items) + cols - 1) // cols)
    ch = int(cell * 9 / 16) + 22
    sheet = Image.new("RGB", (cols * cell, rows * ch), (20, 20, 24))
    d = ImageDraw.Draw(sheet)
    for k, (thumb, cap) in enumerate(items):
        x, y = (k % cols) * cell, (k // cols) * ch
        try:
            with Image.open(thumb) as im:
                im = im.convert("RGB")
                im.thumbnail((cell - 4, ch - 26))
                sheet.paste(im, (x + (cell - im.width) // 2, y + 2))
        except OSError:
            pass
        d.text((x + 4, y + ch - 18), cap, fill=(235, 235, 235))
    out = Path(out)
    sheet.save(out, quality=85)
    return out
