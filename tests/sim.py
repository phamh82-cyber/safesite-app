"""Bộ giả lập chuyển động người / xe để kiểm thử logic mà không cần video."""
from __future__ import annotations

import math

import numpy as np

from app.tracks import Detection

# Độ cao các khớp (tỉ lệ so với chiều cao người) khi đứng; x là độ lệch ngang
STAND = {0: (0.00, 0.94), 1: (-0.02, 0.96), 2: (0.02, 0.96), 3: (-0.05, 0.94), 4: (0.05, 0.94),
         5: (-0.12, 0.82), 6: (0.12, 0.82), 7: (-0.15, 0.65), 8: (0.15, 0.65),
         9: (-0.16, 0.50), 10: (0.16, 0.50), 11: (-0.08, 0.52), 12: (0.08, 0.52),
         13: (-0.08, 0.28), 14: (0.08, 0.28), 15: (-0.08, 0.02), 16: (0.08, 0.02)}
# Quỳ gối: gối chạm đất, thân dựng thẳng
KNEEL = {**STAND, 11: (-0.08, 0.30), 12: (0.08, 0.30), 13: (-0.08, 0.04), 14: (0.08, 0.04),
         15: (-0.08, 0.02), 16: (0.08, 0.02), 5: (-0.12, 0.60), 6: (0.12, 0.60),
         0: (0.00, 0.72), 1: (-0.02, 0.74), 2: (0.02, 0.74), 3: (-0.05, 0.72), 4: (0.05, 0.72),
         7: (-0.15, 0.45), 8: (0.15, 0.45), 9: (-0.10, 0.32), 10: (0.10, 0.32)}
# Ngồi xổm nhặt đồ: hông thấp, thân nghiêng ~40°
SQUAT = {**STAND, 11: (-0.10, 0.28), 12: (0.06, 0.28), 13: (0.10, 0.30), 14: (0.18, 0.30),
         5: (0.05, 0.50), 6: (0.22, 0.50), 0: (0.20, 0.58), 1: (0.19, 0.60), 2: (0.23, 0.60),
         3: (0.16, 0.58), 4: (0.26, 0.58), 7: (0.15, 0.36), 8: (0.28, 0.36),
         9: (0.25, 0.22), 10: (0.32, 0.22)}


def posture_points(post: dict, fx: float, fy: float, H: float, topple: float = 0.0,
                   jitter: float = 0.0, rng=None) -> np.ndarray:
    """Trả về keypoint (17,3). `topple` = góc đổ cả thân quanh bàn chân (độ)."""
    a = math.radians(topple)
    k = np.zeros((17, 3))
    for i, (x, h) in post.items():
        # điểm ở (x, h) so với chân; quay quanh chân theo góc a
        px = x * math.cos(a) + h * math.sin(a)
        ph = -x * math.sin(a) + h * math.cos(a)
        k[i] = (fx + px * H, fy - ph * H, 0.9)
    if jitter and rng is not None:
        k[:, :2] += rng.normal(0, jitter * H, (17, 2))
    return k


def blend(a: dict, b: dict, s: float) -> dict:
    return {i: (a[i][0] * (1 - s) + b[i][0] * s, a[i][1] * (1 - s) + b[i][1] * s) for i in a}


def person_det(tid: int, k: np.ndarray, H: float) -> Detection:
    m = 0.05 * H
    box = np.array([k[:, 0].min() - m, k[:, 1].min() - m, k[:, 0].max() + m, k[:, 1].max() + m])
    return Detection(tid, "person", "person", box, 0.9, k)


def vehicle_det(tid: int, cx: float, bottom: float, w: float, h: float, cls="car") -> Detection:
    return Detection(tid, "vehicle", cls, np.array([cx - w / 2, bottom - h, cx + w / 2, bottom]), 0.9)
