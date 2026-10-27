"""Photo shots in Python: composed base image + subpixel Ken Burns (zoom/pan) via cv2.warpAffine."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from montage.config import get_config
from montage.types import KenBurns

T = 16 / 9


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def load_photo(path: str, max_long: int) -> np.ndarray:
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = max_long / max(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        return np.asarray(im)


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, w), max(1, h)), interpolation=interp)


def compose_fill_blur(img: np.ndarray, cw: int, ch: int) -> np.ndarray:
    """Blurred, darkened cover background + fitted foreground, at canvas size (cw, ch)."""
    import cv2
    cfg = get_config().render
    h, w = img.shape[:2]
    s = max(cw / w, ch / h)
    bg = _resize(img, int(np.ceil(w * s)), int(np.ceil(h * s)))
    y0, x0 = (bg.shape[0] - ch) // 2, (bg.shape[1] - cw) // 2
    bg = bg[y0:y0 + ch, x0:x0 + cw]
    small = _resize(bg, max(1, cw // 4), max(1, ch // 4))  # blur at 1/4 scale (same look, much faster)
    small = cv2.GaussianBlur(small, (0, 0), cfg.blur_sigma / 4)
    bg = _resize(small, cw, ch).astype(np.float32) + cfg.blur_darken * 255
    s2 = min(cw / w, ch / h)
    fw, fh = max(1, int(round(w * s2))), max(1, int(round(h * s2)))
    fg = _resize(img, fw, fh)
    out = np.clip(bg, 0, 255).astype(np.uint8)
    ox, oy = (cw - fw) // 2, (ch - fh) // 2
    out[oy:oy + fh, ox:ox + fw] = fg
    return out


def base_rect(img_w: int, img_h: int, cx: float, cy: float) -> tuple[float, float, float, float]:
    """Largest 16:9 rectangle in the image, centered on (cx, cy) (normalized; -1 = center), clamped."""
    if img_w / img_h > T:
        rh, rw = float(img_h), img_h * T
    else:
        rw, rh = float(img_w), img_w / T
    px = (cx if cx >= 0 else 0.5) * img_w
    py = (cy if cy >= 0 else 0.5) * img_h
    x0 = min(max(px - rw / 2, 0.0), img_w - rw)
    y0 = min(max(py - rh / 2, 0.0), img_h - rh)
    return x0, y0, rw, rh


def frames(photo_path: str, framing: str, kb: KenBurns | None, n: int, W: int, H: int,
           face: tuple[float, float] = (-1.0, -1.0)) -> Iterator[np.ndarray]:
    """Yield n RGB frames (H, W, 3) uint8."""
    import cv2
    cfg = get_config().render
    kb = kb or KenBurns(zoom_from=1.0, zoom_to=1.0, pan_x=0.0, pan_y=0.0)
    if framing == "pan":
        img = load_photo(photo_path, 6 * W)
        h, w = img.shape[:2]
        img = _resize(img, max(W, int(round(w * H / h))), H)
        span = img.shape[1] - W
        for k in range(n):
            s = smoothstep(k / max(n - 1, 1))
            x = span * (s if kb.pan_x >= 0 else 1 - s)
            M = np.float32([[1, 0, x], [0, 1, 0]])
            yield cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
        return
    zmax = max(kb.zoom_from, kb.zoom_to, 1.0)
    if framing == "fill_blur":
        os_ = cfg.fill_blur_oversize
        cw, ch = int(round(W * os_ * zmax)), int(round(H * os_ * zmax))
        img = compose_fill_blur(load_photo(photo_path, 3 * max(cw, ch)), cw, ch)
        rw, rh = cw / os_, ch / os_
        x0, y0 = (cw - rw) / 2, (ch - rh) / 2
    else:
        img = load_photo(photo_path, 3 * W)
        x0, y0, rw, rh = base_rect(img.shape[1], img.shape[0], *face)
        target = W * zmax  # keep ~1 source pixel per output pixel at max zoom (limits aliasing)
        if rw > target * 1.25:
            s = target / rw
            img = _resize(img, int(round(img.shape[1] * s)), int(round(img.shape[0] * s)))
            x0, y0, rw, rh = x0 * s, y0 * s, rw * s, rh * s
    ih, iw = img.shape[:2]
    c0 = np.array([x0 + rw / 2, y0 + rh / 2])
    c1 = c0 + np.array([kb.pan_x * rw, kb.pan_y * rh])
    for k in range(n):
        s = smoothstep(k / max(n - 1, 1))
        z = kb.zoom_from + (kb.zoom_to - kb.zoom_from) * s
        cw_, ch_ = rw / z, rh / z
        cx, cy = c0 + (c1 - c0) * s
        cx = min(max(cx, cw_ / 2), iw - cw_ / 2)
        cy = min(max(cy, ch_ / 2), ih - ch_ / 2)
        sx, sy = cw_ / W, ch_ / H
        M = np.float32([[sx, 0, cx - cw_ / 2 + 0.5 * sx - 0.5], [0, sy, cy - ch_ / 2 + 0.5 * sy - 0.5]])
        yield cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
