"""Lõi phân tích của một camera: nhận danh sách detection (đã có track ID) tại thời
điểm t, cập nhật lịch sử, chiếu xuống nền và chạy các analyzer.
Tách khỏi phần đọc video/mô hình để kiểm thử được bằng dữ liệu giả lập.

An toàn đa luồng: `process` và `set_calibration` dùng chung một khóa, vì hiệu chỉnh được
gọi từ luồng API trong khi luồng camera đang phân tích.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from .analyzers.collision import CollisionAnalyzer
from .analyzers.fall import FallAnalyzer, inherit_fall_state
from .analyzers.hazard import HazardAnalyzer
from .analyzers.height_fall import HeightFallAnalyzer
from .analyzers.objects import FallingObjectAnalyzer
from .config import CLASS_HEIGHT_M, Settings
from .ego import EgoMotion
from .geometry import GroundMapper, vehicle_footprint_img
from .tracks import L_ANK, R_ANK, Detection, Obs, TrackManager


class Engine:
    def __init__(self, settings: Settings, img_w: int, img_h: int):
        self.settings = settings
        self.w, self.h = img_w, img_h
        self._lock = threading.RLock()
        self.mapper = GroundMapper(img_w, img_h, settings.pipeline.focal_ratio)
        self.tm = TrackManager(keep_lost=5.0)
        self.fall = FallAnalyzer(settings.fall, img_w, img_h)
        self.collision = CollisionAnalyzer(settings.collision, img_w, img_h)
        self.height_fall = HeightFallAnalyzer(settings.height_fall)
        self.objects = FallingObjectAnalyzer(settings.objects, img_w, img_h)
        self.hazard = HazardAnalyzer(settings.hazard, img_w, img_h)
        self.ego = EgoMotion(img_w, img_h)
        self._map_mode = self.mapper.mode
        self._map_fit = None

    @property
    def shaking(self) -> bool:
        return self.ego.level > self.settings.pipeline.shake_level

    # ------------------------------------------------------------------
    def _foot(self, d: Detection) -> np.ndarray:
        x1, y1, x2, y2 = d.box
        if d.kind == "person" and d.kpts is not None:
            ank = [d.kpts[i, :2] for i in (L_ANK, R_ANK) if d.kpts[i, 2] >= 0.4]
            if ank and (y2 - y1) > 1.3 * (x2 - x1):   # chỉ dùng mắt cá khi người đứng
                return np.array([np.mean([a[0] for a in ank]), max(a[1] for a in ank)])
        return np.array([(x1 + x2) / 2, y2], float)

    def _border(self, box, m: int = 4) -> bool:
        return box[1] <= m or box[3] >= self.h - m or box[0] <= m or box[2] >= self.w - m

    def _size_sample(self, d: Detection) -> None:
        x1, y1, x2, y2 = d.box
        if self._border(d.box) or self.mapper.H is not None:
            return
        h, w = y2 - y1, x2 - x1
        if d.kind == "person" and h > 2.2 * w:
            self.mapper.add_size_sample(y2, h, CLASS_HEIGHT_M["person"])
        elif d.kind == "vehicle" and d.cls_name in CLASS_HEIGHT_M:
            self.mapper.add_size_sample(y2, h, CLASS_HEIGHT_M[d.cls_name])

    def _reproject_all(self) -> None:
        for tr in self.tm.tracks.values():
            hist = list(tr.hist)
            tr.hist.clear()
            for o in hist:
                o.ground = self.mapper.to_ground(o.foot_img - o.shift)[0]
                if o.footprint is not None:
                    o.footprint = self.mapper.to_ground(vehicle_footprint_img(o.box) - o.shift)
                o.vel = np.zeros(2)
                tr.add(o)

    # ------------------------------------------------------------------
    def process(self, t: float, dets: List[Detection], frame: Optional[np.ndarray] = None,
                hazards: Optional[list] = None) -> List[dict]:
        """frame: khung BGR gốc (cần cho bù rung, vật rơi, cháy). hazards: kết quả mô hình cháy/khói
        ở khung này [(kind, conf, box)], None nếu mô hình không chạy ở khung này."""
        with self._lock:
            return self._process(t, dets, frame, hazards)

    def _process(self, t, dets, frame, hazards) -> List[dict]:
        p = self.settings.pipeline
        gray = None
        if frame is not None:
            gray = self.ego.update(frame, [d.box for d in dets], estimate=p.stabilize)
        shift = self.ego.off.copy()
        seen = set()
        new_people = []
        for d in dets:
            self._size_sample(d)
        fit = (self.mapper.mode, self.mapper._fit)
        if fit != (self._map_mode, self._map_fit):
            self._map_mode, self._map_fit = fit
            self._reproject_all()
        for d in dets:
            key = ("P" if d.kind == "person" else "V") + str(d.track_id)
            seen.add(key)
            tr, is_new = self.tm.get_or_create(key, d.kind, d.cls_name)
            foot = self._foot(d)
            fp = None
            if d.kind == "vehicle":
                fp = self.mapper.to_ground(vehicle_footprint_img(d.box) - shift)
            obs = Obs(t=t, box=np.asarray(d.box, float), conf=d.conf, kpts=d.kpts,
                      foot_img=foot, ground=self.mapper.to_ground(foot - shift)[0], footprint=fp,
                      shift=shift)
            tr.add(obs)
            if is_new and d.kind == "person":
                new_people.append(tr)
        self.tm.mark_unseen(seen, t)
        for tr in new_people:
            inherit_fall_state(self.tm, tr)
        fall_events = self.fall.step(self.tm, t)
        col_events = self.collision.step(self.tm, t, fall_events)
        hf_events = self.height_fall.step(self.tm, t, fall_events)
        shaking = self.shaking
        stab = self.ego.stabilize(gray) if gray is not None else None
        obj_events = self.objects.step(self.tm, t, stab, shaking, shift)
        hz_events = self.hazard.step(self.tm, t, frame, hazards, shaking)
        # một vụ ngã từ trên cao không báo thêm "té ngã" thường cho cùng người
        fall_events = [e for e in fall_events if not self._dup_fall(e, t)]
        return fall_events + col_events + hf_events + obj_events + hz_events

    def _dup_fall(self, ev: dict, t: float) -> bool:
        if ev.get("type") != "FALL" or not ev.get("track_ids"):
            return False
        tid = ev["track_ids"][0]
        ids = {tid}
        tr = self.tm.tracks.get(tid)
        if tr is not None and tr.state.get("inherited_from"):
            ids.add(tr.state["inherited_from"])
        return self.height_fall.recent_any(ids, t)

    def set_calibration(self, img_pts, ground_pts) -> None:
        with self._lock:
            if img_pts:
                self.mapper.set_homography(img_pts, ground_pts)
            else:
                self.mapper.clear_homography()
            self._map_mode = self.mapper.mode
            self._reproject_all()
