"""ByteTrack: dùng BYTETracker của ultralytics (độc lập với model.track nên mỗi camera có tracker riêng).

Nếu không import được thì rơi về tracker IoU đơn giản để hệ thống vẫn chạy.
ID do ultralytics cấp dùng chung 1 bộ đếm => duy nhất trên toàn hệ thống.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np


class _Dets:
    """Giả lập đối tượng Boxes mà BYTETracker.update() cần: conf, xywh, cls, len, indexing."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return _Dets(self.xyxy[i], self.conf[i], self.cls[i])


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class _IouTracker:
    _ids = itertools.count(1_000_000)

    def __init__(self, iou_thr=0.3, max_age=30):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks = {}  # id -> [box, age]

    def update(self, d: _Dets):
        out, used = [], set()
        for i in range(len(d)):
            box = d.xyxy[i]
            best, best_iou = None, self.iou_thr
            for tid, (tb, _) in self.tracks.items():
                if tid in used:
                    continue
                v = _iou(box, tb)
                if v > best_iou:
                    best, best_iou = tid, v
            if best is None:
                best = next(self._ids)
            used.add(best)
            self.tracks[best] = [box, 0]
            out.append([*box, best, d.conf[i], d.cls[i], i])
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.max_age:
                    del self.tracks[tid]
        return np.asarray(out, dtype=np.float32).reshape(-1, 8)


class ByteTrackWrapper:
    def __init__(self, frame_rate: int = 30):
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1,
                                   new_track_thresh=0.25, track_buffer=30,
                                   match_thresh=0.8, fuse_score=True)
            self._t = BYTETracker(args, frame_rate=frame_rate)
            self.backend = "bytetrack"
        except Exception as e:  # pragma: no cover
            print(f"[tracker] BYTETracker không khả dụng ({e}); dùng IoU tracker.")
            self._t = _IouTracker()
            self.backend = "iou"

    def update(self, xyxy, conf, cls=None):
        """Trả về mảng (M, 8): x1,y1,x2,y2,track_id,score,cls,det_index."""
        n = len(conf)
        if cls is None:
            cls = np.zeros(n, dtype=np.float32)
        d = _Dets(xyxy if n else np.zeros((0, 4)), conf, cls)
        try:
            out = self._t.update(d) if self.backend == "iou" else self._t.update(d, None)
        except TypeError:
            out = self._t.update(d)
        out = np.asarray(out, dtype=np.float32)
        return out.reshape(-1, 8) if out.size else np.zeros((0, 8), dtype=np.float32)
