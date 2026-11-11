"""Lưu lịch sử theo thời gian của từng đối tượng (người / phương tiện)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Chỉ số keypoint theo chuẩn COCO-17
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 5, 6, 11, 12, 15, 16
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


@dataclass
class Detection:
    track_id: int
    kind: str                     # "person" | "vehicle"
    cls_name: str
    box: np.ndarray               # x1, y1, x2, y2 (pixel)
    conf: float
    kpts: Optional[np.ndarray] = None   # (17, 3): x, y, conf


@dataclass
class Obs:
    t: float
    box: np.ndarray
    conf: float
    kpts: Optional[np.ndarray]
    foot_img: np.ndarray
    ground: np.ndarray            # vị trí trên nền (m)
    footprint: Optional[np.ndarray] = None  # đa giác trên nền (xe)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    shift: np.ndarray = field(default_factory=lambda: np.zeros(2))   # độ lệch ảnh do rung camera (px)


class Track:
    def __init__(self, tid: str, kind: str, cls_name: str):
        self.id, self.kind, self.cls_name = tid, kind, cls_name
        self.hist: deque[Obs] = deque(maxlen=200)
        self.lost = False
        self.lost_since: Optional[float] = None
        self.state: dict = {}     # trạng thái riêng của các analyzer

    # ----- truy vấn -----
    @property
    def last(self) -> Obs:
        return self.hist[-1]

    @property
    def last_t(self) -> float:
        return self.hist[-1].t

    def since(self, t0: float) -> List[Obs]:
        return [o for o in self.hist if o.t >= t0]

    def at_or_before(self, t: float) -> Optional[Obs]:
        for o in reversed(self.hist):
            if o.t <= t:
                return o
        return None

    def velocity(self) -> np.ndarray:
        return self.hist[-1].vel if self.hist else np.zeros(2)

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity()))

    def max_speed_since(self, t0: float) -> float:
        return max((float(np.linalg.norm(o.vel)) for o in self.since(t0)), default=0.0)

    # ----- cập nhật -----
    def add(self, obs: Obs, window: float = 0.6) -> None:
        self.hist.append(obs)
        if len(self.hist) >= 2:   # FPS thấp: nới cửa sổ để luôn có ≥3 mẫu (nếu không vận tốc = 0)
            window = min(max(window, 3.2 * (obs.t - self.hist[-2].t)), 1.5)
        pts = [o for o in self.hist if o.t >= obs.t - window]
        if len(pts) >= 3:
            ts = np.array([o.t for o in pts]) - obs.t
            g = np.array([o.ground for o in pts])
            if np.ptp(ts) > 1e-3:
                vx = np.polyfit(ts, g[:, 0], 1)[0]
                vy = np.polyfit(ts, g[:, 1], 1)[0]
                v = np.array([vx, vy])
                prev = self.hist[-2].vel if len(self.hist) >= 2 else v
                obs.vel = 0.6 * v + 0.4 * prev   # EMA làm mượt
        self.lost, self.lost_since = False, None


class TrackManager:
    """Giữ track đã mất tới `keep_lost` giây; nối lại ID cho người vừa ngã
    (detector thường mất người ở tư thế nằm → tracker cấp ID mới)."""

    def __init__(self, keep_lost: float = 5.0):
        self.tracks: Dict[str, Track] = {}
        self.keep_lost = keep_lost

    def get_or_create(self, key: str, kind: str, cls_name: str) -> tuple[Track, bool]:
        tr = self.tracks.get(key)
        if tr is None:
            tr = Track(key, kind, cls_name)
            self.tracks[key] = tr
            return tr, True
        return tr, False

    def mark_unseen(self, seen: set, t: float) -> None:
        for k, tr in list(self.tracks.items()):
            if k in seen:
                continue
            if not tr.lost:
                tr.lost, tr.lost_since = True, t
            elif t - tr.lost_since > self.keep_lost:
                del self.tracks[k]

    def find_lost_person_near(self, foot_img: np.ndarray, t: float, radius_px: float,
                              max_age: float = 3.0, allow_normal: bool = False) -> Optional[Track]:
        best, best_d = None, radius_px
        for tr in self.tracks.values():
            if tr.kind != "person" or not tr.lost or t - tr.lost_since > max_age:
                continue
            fs = tr.state.get("fall")
            if fs is None or (fs.get("phase") in (None, "NORMAL") and not allow_normal):
                continue
            if allow_normal and fs.get("phase") != "NORMAL":
                continue
            d = float(np.linalg.norm(tr.last.foot_img - foot_img))
            if d < best_d:
                best, best_d = tr, d
        return best

    def persons(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.kind == "person"]

    def vehicles(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.kind == "vehicle"]
