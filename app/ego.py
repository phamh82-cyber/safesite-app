"""Ước lượng rung / dịch chuyển toàn cảnh của camera (điện thoại cầm tay hoặc kẹp lỏng).

Mỗi khung: lấy điểm đặc trưng trên nền (bỏ vùng người/xe đang chuyển động), theo dõi bằng
Lucas–Kanade, rồi lấy dịch chuyển *đồng thuận* của các điểm. Nếu quá ít điểm cùng dịch chuyển
(cảnh toàn vật thể chuyển động, thiếu vân, ban đêm) thì coi như không đo được và không bù.

Kết quả:
- `off`   : độ lệch tích lũy có "rò" (px, ảnh gốc). Tọa độ ổn định = tọa độ ảnh − off.
            Rò dần về 0 để sai số tích lũy và việc xoay camera chậm không làm lệch mãi.
- `level` : mức rung hiện tại (tỉ lệ chiều rộng ảnh mỗi khung, làm mượt).
Chỉ bù tịnh tiến; xoay/zoom nhỏ bỏ qua.
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np


class EgoMotion:
    def __init__(self, img_w: int, img_h: int, work_width: int = 320, leak: float = 0.97):
        self.w, self.h = img_w, img_h
        self.s = min(1.0, work_width / img_w)
        self.size = (max(32, int(round(img_w * self.s))), max(32, int(round(img_h * self.s))))
        self.leak = leak
        self.prev: Optional[np.ndarray] = None
        self.off = np.zeros(2)
        self.level = 0.0
        self.ok = False          # khung gần nhất có đo được hay không

    # ------------------------------------------------------------------
    def to_gray_small(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    def update(self, frame: np.ndarray, boxes: Iterable = (), estimate: bool = True) -> np.ndarray:
        """Cập nhật theo khung mới; trả về ảnh xám thu nhỏ (chưa ổn định)."""
        g = self.to_gray_small(frame)
        d = np.zeros(2)
        self.ok = False
        if estimate and self.prev is not None:
            d = self._shift(self.prev, g, boxes)
        self.prev = g
        self.level = 0.5 * self.level + 0.5 * float(np.linalg.norm(d)) / self.w
        self.off = self.leak * self.off + d
        return g

    def _shift(self, prev: np.ndarray, cur: np.ndarray, boxes: Iterable) -> np.ndarray:
        mask = np.full(prev.shape, 255, np.uint8)
        for b in boxes:
            x1, y1, x2, y2 = (np.asarray(b, float) * self.s).astype(int)
            pad = 4
            cv2.rectangle(mask, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 0, -1)
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=6, mask=mask)
        if p0 is None or len(p0) < 20:
            return np.zeros(2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
        if p1 is None:
            return np.zeros(2)
        good = st.ravel() == 1
        if good.sum() < 20:
            return np.zeros(2)
        flow = (p1 - p0).reshape(-1, 2)[good]
        med = np.median(flow, axis=0)
        inl = np.linalg.norm(flow - med, axis=1) < 1.5
        if inl.sum() < max(15, 0.4 * len(flow)):     # không có dịch chuyển đồng thuận
            return np.zeros(2)
        self.ok = True
        return flow[inl].mean(axis=0) / self.s

    # ------------------------------------------------------------------
    def stabilize(self, gray_small: np.ndarray) -> np.ndarray:
        """Dịch ảnh xám thu nhỏ ngược lại độ lệch hiện tại, để nền đứng yên giữa các khung."""
        if not self.off.any():
            return gray_small
        m = np.float32([[1, 0, -self.off[0] * self.s], [0, 1, -self.off[1] * self.s]])
        return cv2.warpAffine(gray_small, m, self.size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
