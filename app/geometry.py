"""Chiếu điểm ảnh xuống mặt phẳng nền (đơn vị mét).

Ba chế độ, theo thứ tự ưu tiên:
1. homography  – người dùng hiệu chỉnh 4 điểm trên nền (chính xác nhất).
2. auto-perspective – tự ước lượng từ chiều cao người/xe: với camera pinhole nhìn
   nền phẳng, chiều cao ảnh h_px của vật cao H (m) đứng tại hàng v thỏa
   h_px / H = (v - v_h) / Hc  (v_h: đường chân trời, Hc: độ cao camera).
   Hồi quy tuyến tính h_px/H theo v cho ra v_h và Hc, từ đó
   Z = f·Hc/(v - v_h),  X = (u - cx)·Hc/(v - v_h).
3. auto-scale – chưa đủ mẫu: tỉ lệ m/px cố định (kém chính xác nhất).
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import cv2
import numpy as np


class GroundMapper:
    def __init__(self, img_w: int, img_h: int, focal_ratio: float = 0.9):
        self.w, self.h = img_w, img_h
        self.f = focal_ratio * img_w
        self.cx = img_w / 2
        self.H: Optional[np.ndarray] = None
        self.samples: deque = deque(maxlen=400)
        self._fit: Optional[tuple] = None  # (v_h, Hc)
        self._scale: float = 1.7 / (0.3 * img_h)
        self._dirty = 0

    # ---------------- cấu hình ----------------
    @property
    def mode(self) -> str:
        if self.H is not None:
            return "homography"
        return "auto-perspective" if self._fit else "auto-scale"

    def set_homography(self, img_pts: Sequence[Sequence[float]],
                       ground_pts: Sequence[Sequence[float]]) -> None:
        src = np.asarray(img_pts, np.float32)
        dst = np.asarray(ground_pts, np.float32)
        if len(src) == 4:
            self.H = cv2.getPerspectiveTransform(src, dst)
        else:
            self.H, _ = cv2.findHomography(src, dst)

    def clear_homography(self) -> None:
        self.H = None

    def add_size_sample(self, v_bottom: float, h_px: float, h_m: float) -> None:
        """Mẫu chiều cao của người đứng thẳng / xe để tự hiệu chỉnh."""
        if h_px < 12 or v_bottom <= 0:
            return
        self.samples.append((float(v_bottom), float(h_px) / h_m))
        self._dirty += 1
        every = 10 if (self._fit is None or len(self.samples) < 150) else 100
        if self._dirty >= every:
            self._dirty = 0
            self._refit()

    def _refit(self) -> None:
        arr = np.asarray(self.samples)
        self._scale = float(np.median(1.0 / arr[:, 1]))
        if len(arr) < 12 or np.ptp(arr[:, 0]) < 0.12 * self.h:
            self._fit = None
            return
        a, b = np.polyfit(arr[:, 0], arr[:, 1], 1)
        if a <= 1e-6:
            self._fit = None
            return
        v_h, hc = -b / a, 1.0 / a
        if not (0.5 <= hc <= 40) or v_h >= arr[:, 0].min() - 5:
            self._fit = None
            return
        self._fit = (float(v_h), float(hc))

    # ---------------- chiếu ----------------
    def to_ground(self, pts) -> np.ndarray:
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.H is not None:
            q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.H).reshape(-1, 2)
            return q
        if self._fit:
            v_h, hc = self._fit
            dv = np.maximum(p[:, 1] - v_h, 2.0)
            z = self.f * hc / dv
            x = (p[:, 0] - self.cx) * hc / dv
            return np.stack([x, z], 1)
        return p * self._scale

    def info(self) -> dict:
        d = {"mode": self.mode, "samples": len(self.samples)}
        if self._fit:
            d["horizon_v"], d["camera_height_m"] = round(self._fit[0], 1), round(self._fit[1], 2)
        return d


# ---------------- hình học trên mặt nền ----------------
def point_polygon_distance(pt, poly: np.ndarray) -> float:
    d = cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2),
                             (float(pt[0]), float(pt[1])), True)
    return max(0.0, -d)


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    try:
        area, _ = cv2.intersectConvexConvex(a32, b32)
        if area > 1e-6:
            return 0.0
    except cv2.error:
        pass
    return float(min(min(point_polygon_distance(p, b) for p in a),
                     min(point_polygon_distance(p, a) for p in b)))


def closest_point_on_polygon(pt, poly: np.ndarray) -> np.ndarray:
    pt = np.asarray(pt, float)
    if point_polygon_distance(pt, poly) == 0.0:
        return pt
    best, best_d = None, 1e18
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        c = a + t * ab
        d = np.sum((pt - c) ** 2)
        if d < best_d:
            best, best_d = c, d
    return best


def box_iou_any(a, b) -> bool:
    """Hai bbox trên ảnh có giao nhau không."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def vehicle_footprint_img(box, band: float = 0.35) -> np.ndarray:
    """35% phần dưới bbox xe – xấp xỉ vùng bánh xe chạm nền."""
    x1, y1, x2, y2 = box
    yt = y2 - band * (y2 - y1)
    return np.array([[x1, y2], [x2, y2], [x2, yt], [x1, yt]], float)
