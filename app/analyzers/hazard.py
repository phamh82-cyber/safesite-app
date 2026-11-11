"""Cháy / khói (FIRE_SMOKE).

Nguồn phát hiện:
  • Mô hình YOLO fine-tune (kaggle/train_fire_smoke.ipynb) – chạy mỗi `every` khung ở pipeline,
    kết quả truyền vào `step(hazards=[(kind, conf, box), ...])`, kind ∈ {"fire", "smoke"}.
  • Chưa có mô hình: bộ lọc màu lửa (HSV) + yêu cầu vùng màu *nhấp nháy* và loại vùng người/xe.
    Chỉ nhận ra lửa, không nhận ra khói; dễ báo nhầm hơn mô hình (đèn hàn, đèn pha, hoàng hôn).

Một phát hiện đơn lẻ không đủ để báo: cần `min_hits` phát hiện cùng vị trí trong `window` giây.
Lửa: mức "critical"; khói: "warning".
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..config import HazardSettings
from ..tracks import TrackManager


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


def _same_place(a, b) -> bool:
    if _iou(a, b) >= 0.1:
        return True
    ca, cb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])) < 0.5 * size


class HazardAnalyzer:
    KINDS = ("fire", "smoke")

    def __init__(self, cfg: HazardSettings, img_w: int, img_h: int):
        self.cfg, self.w, self.h = cfg, img_w, img_h
        self.model_active = False            # pipeline đặt True khi đã nạp được mô hình
        self.obs: Dict[str, deque] = {k: deque() for k in self.KINDS}
        self.last_alert: Dict[str, float] = {}
        self.last_boxes: list = []           # để vẽ: (kind, conf, box)
        self._boxes_t = -1e9
        self._prev_mask: Optional[np.ndarray] = None
        self._n = 0
        self._k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ------------------------------------------------------------------
    def step(self, tm: TrackManager, t: float, frame: Optional[np.ndarray], hazards: Optional[list],
             shaking: bool = False) -> List[dict]:
        c = self.cfg
        if not c.enabled:
            return []
        if hazards is not None:                           # mô hình vừa chạy ở khung này
            self.last_boxes = [(k, float(cf), np.asarray(b, float)) for k, cf, b in hazards]
            self._boxes_t = t
            for kind, conf, box in self.last_boxes:
                if kind in self.obs:
                    self.obs[kind].append((t, box, conf))
        elif not self.model_active and c.color_fallback and frame is not None:
            self._n += 1
            if self._n % max(1, c.every) == 0 and not shaking:
                det = self._color_fire(frame, tm)
                self.last_boxes = [("fire", det[1], det[0])] if det else []
                self._boxes_t = t
                if det:
                    self.obs["fire"].append((t, det[0], det[1]))
        if t - self._boxes_t > 1.0:
            self.last_boxes = []
        events: List[dict] = []
        for kind in self.KINDS:
            events += self._check(kind, t)
        return events

    def _check(self, kind: str, t: float) -> List[dict]:
        c, dq = self.cfg, self.obs[kind]
        while dq and dq[0][0] < t - c.window:
            dq.popleft()
        if len(dq) < c.min_hits or t - self.last_alert.get(kind, -1e9) < c.cooldown:
            return []
        newest = dq[-1]
        hits = [o for o in dq if _same_place(o[1], newest[1])]
        if len(hits) < c.min_hits:
            return []
        self.last_alert[kind] = t
        conf = float(np.mean([o[2] for o in hits]))
        box = newest[1]
        area = (box[2] - box[0]) * (box[3] - box[1]) / (self.w * self.h)
        return [{"key": f"FIRE_SMOKE:{kind}:{t:.2f}", "type": "FIRE_SMOKE",
                 "severity": "critical" if kind == "fire" else "warning",
                 "confidence": round(min(conf, 0.99), 2), "t": t, "track_ids": [],
                 "boxes": [box.tolist()],
                 "details": {"kind": kind, "source": "model" if self.model_active else "color",
                             "hits": len(hits), "area_pct": round(100 * area, 1)}}]

    # ------------------------------------------------------------------
    def _color_fire(self, frame: np.ndarray, tm: TrackManager):
        c = self.cfg
        s = min(1.0, 320 / self.w)
        small = cv2.resize(frame, (max(32, int(self.w * s)), max(32, int(self.h * s))),
                           interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hh, ss, vv = cv2.split(hsv)
        mask = (((hh <= 28) | (hh >= 172)) & (ss >= 110) & (vv >= 200)).astype(np.uint8) * 255
        for tr in tm.tracks.values():                     # áo phản quang, máy móc cam/đỏ không phải lửa
            if tr.lost:
                continue
            x1, y1, x2, y2 = (tr.last.box * s).astype(int)
            cv2.rectangle(mask, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 0, -1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k3)
        prev, self._prev_mask = self._prev_mask, mask
        area = cv2.countNonZero(mask) / mask.size
        if prev is None or not (c.color_min_area <= area <= c.color_max_area):
            return None
        union = cv2.countNonZero(cv2.bitwise_or(mask, prev))
        flicker = 1.0 - cv2.countNonZero(cv2.bitwise_and(mask, prev)) / max(union, 1)
        if flicker < c.color_flicker:                     # đứng yên: biển báo, cọc tiêu, đèn
            return None
        ys, xs = np.nonzero(mask)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float) / s
        return box, float(min(0.6, 0.35 + 0.4 * flicker))
