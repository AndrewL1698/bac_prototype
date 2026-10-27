"""YuNet face detection on analysis frames (one detector per thread)."""

from __future__ import annotations

import threading

import numpy as np

from montage.config import get_config
from montage.log import get_logger

log = get_logger("faces")
_local = threading.local()
_failed = False


def _detector():
    global _failed
    if _failed:
        return None
    det = getattr(_local, "det", None)
    if det is None:
        try:
            import cv2
            from montage import weights
            det = cv2.FaceDetectorYN.create(str(weights.yunet_path()), "", (320, 320), get_config().features.face_score)
            _local.det = det
        except Exception as e:  # noqa: BLE001  optional component
            log.warning("YuNet unavailable (%s); faces disabled", e)
            _failed = True
            return None
    return det


def frame_faces(path: str) -> tuple[int, float, float, float]:
    """(count, largest face area fraction, union-box center x, y normalized; -1 if no face)."""
    import cv2
    det = _detector()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if det is None or img is None:
        return 0, 0.0, -1.0, -1.0
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return 0, 0.0, -1.0, -1.0
    x0, y0 = faces[:, 0].min(), faces[:, 1].min()
    x1, y1 = (faces[:, 0] + faces[:, 2]).max(), (faces[:, 1] + faces[:, 3]).max()
    area = float((faces[:, 2] * faces[:, 3]).max() / (w * h))
    cx = float(np.clip((x0 + x1) / 2 / w, 0, 1))
    cy = float(np.clip((y0 + y1) / 2 / h, 0, 1))
    return int(len(faces)), area, cx, cy
