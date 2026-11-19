"""Quản lý vòng đời track, lịch sử toạ độ, vận tốc v_y và sinh Candidate (nghi ngờ)."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.analyzers.yolo_detector import calc_torso_angle


@dataclass
class Track:
    id: int
    kind: str                     # "person" | "vehicle"
    box: tuple = (0, 0, 0, 0)
    conf: float = 0.0
    name: str = ""
    angle: Optional[float] = None
    lying_frames: int = 0
    vy: float = 0.0               # vận tốc dọc (chiều cao người / giây), dương = đi xuống
    first_seen: float = 0.0
    last_seen: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=90))   # (ts, cx, cy)

    @property
    def height(self):
        return max(1.0, self.box[3] - self.box[1])


@dataclass
class Candidate:
    reason: str                   # fall | vehicle_proximity | fire | scan
    key: str                      # khoá chống spam (reason:track_id)
    track_ids: list
    boxes: list
    score: float                  # độ mạnh của bằng chứng hình học 0..1
    detail: str = ""


def box_gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


class TrackManager:
    STALE_S = 2.0

    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self.fires: list = []

    def persons(self):
        return [t for t in self.tracks.values() if t.kind == "person"]

    def vehicles(self):
        return [t for t in self.tracks.values() if t.kind == "vehicle"]

    def clear(self):
        self.tracks.clear()
        self.fires = []

    def update(self, ts: float, s, persons, vehicles, fires):
        """persons/vehicles: list of (track_id, Det). Cập nhật trạng thái track."""
        for kind, items in (("person", persons), ("vehicle", vehicles)):
            for tid, det in items:
                tid = int(tid)
                t = self.tracks.get(tid)
                if t is None:
                    t = self.tracks[tid] = Track(tid, kind, first_seen=ts)
                t.box, t.conf, t.name, t.last_seen = det.box, det.conf, det.name, ts
                cx, cy = (det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2
                t.history.append((ts, cx, cy))
                if kind == "person":
                    t.angle = calc_torso_angle(det.kpts, det.box)
                    if t.angle is not None and t.angle > s.lying_angle_deg:
                        t.lying_frames += 1
                    else:
                        t.lying_frames = 0
                    t.vy = self._vy(t)
        for tid in [k for k, t in self.tracks.items() if ts - t.last_seen > self.STALE_S]:
            del self.tracks[tid]
        self.fires = list(fires)

    @staticmethod
    def _vy(t: Track) -> float:
        h = t.history
        if len(h) < 2:
            return 0.0
        ts1, _, cy1 = h[-1]
        for ts0, _, cy0 in h:
            if ts1 - ts0 <= 0.5:
                dt = ts1 - ts0
                return (cy1 - cy0) / dt / t.height if dt > 0.05 else 0.0
        return 0.0

    def candidates(self, s, now: float) -> list[Candidate]:
        out: list[Candidate] = []
        live = lambda t: now - t.last_seen < 0.5
        veh = [v for v in self.vehicles() if live(v)]
        for p in self.persons():
            if not live(p):
                continue
            fast_drop = p.vy > 1.5 and (p.angle or 0) > 0.6 * s.lying_angle_deg
            if p.lying_frames >= s.fall_min_frames or fast_drop:
                score = 0.5
                if p.lying_frames >= 3 * s.fall_min_frames:
                    score += 0.25
                if (p.angle or 0) >= s.lying_angle_deg + 15:
                    score += 0.25
                out.append(Candidate("fall", f"fall:{p.id}", [p.id], [p.box], score,
                                     f"angle={p.angle:.0f}" if p.angle is not None else "bbox"))
            for v in veh:
                gap = box_gap(p.box, v.box) / p.height
                if gap < s.safe_distance:
                    score = 0.8 if gap <= 0 else 0.4 + 0.4 * (1 - gap / max(s.safe_distance, 1e-3))
                    out.append(Candidate("vehicle_proximity", f"vehicle_proximity:{p.id}",
                                         [p.id, v.id], [p.box, v.box], score,
                                         f"gap={gap:.2f}h {v.name}"))
        for f in self.fires:
            out.append(Candidate("fire", "fire", [], [f.box], float(f.conf), f.name))
        return out
