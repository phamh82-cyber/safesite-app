"""Vật rơi (OBJECT_FALL) và người bị vật rơi đập trúng (STRUCK_BY).

Quy trình trên ảnh xám thu nhỏ đã ổn định (bù rung camera):
  1. Trừ nền MOG2 → mặt nạ tiền cảnh → các vệt (blob) kích thước vừa phải.
  2. Ghép vệt qua các khung (gần nhất theo vị trí dự đoán) thành quỹ đạo.
  3. Quỹ đạo là "vật rơi" khi: đi xuống đủ xa, chủ yếu theo phương đứng, gần như đơn điệu,
     đang tăng tốc (không phải băng chuyền / vật bị mang đi) và không phải người/xe đã biết.
  4. Vật rơi có tâm lọt vào bbox người (mở rộng contact_margin) ⇒ "tiếp xúc"; trong
     confirm_window giây người ngã / mất dấu ⇒ STRUCK_BY.

Giới hạn: vật nhỏ hơn min_area, vật cùng màu nền, vật rơi sau lưng người (bbox trùng nhưng ở xa) có thể
bị bỏ sót hoặc tính nhầm; mưa, tuyết, côn trùng sát ống kính được lọc bằng diện tích và số vệt tối đa.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..config import ObjectSettings
from ..tracks import Track, TrackManager

FALLING_PHASES = ("DESCENDING", "LYING", "FALLEN")


@dataclass
class _Blob:
    id: int
    hist: deque = field(default_factory=lambda: deque(maxlen=40))   # (t, cx, cy, w, h)
    miss: int = 0
    known: int = 0                  # số mẫu chồng lên người/xe đã biết
    n: int = 0
    reported: bool = False
    contacted: set = field(default_factory=set)


class FallingObjectAnalyzer:
    def __init__(self, cfg: ObjectSettings, img_w: int, img_h: int):
        self.cfg, self.w, self.h = cfg, img_w, img_h
        self.s = min(1.0, cfg.work_width / img_w)
        self.size = (max(32, int(round(img_w * self.s))), max(32, int(round(img_h * self.s))))
        self.mog = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32, detectShadows=False)
        self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.blobs: Dict[int, _Blob] = {}
        self.next_id = 1
        self.t0: Optional[float] = None
        self.last_report = -1e9
        self.contacts: List[dict] = []
        self.last_struck: Dict[str, float] = {}
        self.overlay: List[dict] = []         # để vẽ: {"box": [x1,y1,x2,y2] (ảnh gốc), "v": ...}

    # ------------------------------------------------------------------
    def _known_boxes(self, tm: TrackManager, shift: np.ndarray):
        """bbox người/xe trong hệ tọa độ ảnh làm việc đã ổn định."""
        sh = np.tile(shift, 2)
        people, vehicles = [], []
        for tr in tm.tracks.values():
            if tr.lost:
                continue
            b = (tr.last.box - sh) * self.s
            (people if tr.kind == "person" else vehicles).append((tr, b))
        return people, vehicles

    @staticmethod
    def _inside_frac(cx, cy, w, h, box) -> float:
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        iw = max(0.0, min(x2, box[2]) - max(x1, box[0]))
        ih = max(0.0, min(y2, box[3]) - max(y1, box[1]))
        return iw * ih / max(w * h, 1e-6)

    def _to_raw(self, box_work, shift) -> list:
        x1, y1, x2, y2 = (np.asarray(box_work, float) / self.s)
        return [float(x1 + shift[0]), float(y1 + shift[1]), float(x2 + shift[0]), float(y2 + shift[1])]

    # ------------------------------------------------------------------
    def step(self, tm: TrackManager, t: float, gray: Optional[np.ndarray], shaking: bool,
             shift: np.ndarray) -> List[dict]:
        c = self.cfg
        self.overlay = []
        if not c.enabled or gray is None:
            return []
        if self.t0 is None:
            self.t0 = t
        fg = self.mog.apply(gray)                 # luôn học nền, kể cả khi bỏ khung
        events: List[dict] = []
        people, vehicles = self._known_boxes(tm, shift)
        usable = t - self.t0 >= c.warmup and not shaking
        if usable:
            _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
            fg = cv2.dilate(cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k3), self.k3)
            if cv2.countNonZero(fg) > c.max_fg_ratio * fg.size:
                usable = False                    # đổi sáng / rung toàn cảnh
        if usable:
            cand = self._candidates(fg)
            if cand is None:                      # quá nhiều vệt (mưa, nhiễu)
                usable = False
        if not usable:
            self.blobs.clear()
        else:
            events += self._track(t, cand, people, vehicles, shift)
        events += self._contacts(tm, t, shift)
        return events

    # ------------------------------------------------------------------
    def _candidates(self, fg: np.ndarray):
        c = self.cfg
        n, _, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
        area = self.size[0] * self.size[1]
        lo, hi = c.min_area * area, c.max_area * area
        cand = [(float(cents[i][0]), float(cents[i][1]), float(stats[i][2]), float(stats[i][3]))
                for i in range(1, n) if lo <= stats[i][4] <= hi]
        return None if len(cand) > c.max_blobs else cand

    def _track(self, t, cand, people, vehicles, shift) -> List[dict]:
        c, events = self.cfg, []
        gate = c.gate * self.size[1]
        blobs = list(self.blobs.values())
        pairs = []
        for bi, b in enumerate(blobs):
            tl, cx, cy = b.hist[-1][:3]
            vx = vy = 0.0
            if len(b.hist) >= 2:
                t0, x0, y0 = b.hist[-2][:3]
                dt = max(tl - t0, 1e-3)
                vx, vy = (cx - x0) / dt, (cy - y0) / dt
            px, py = cx + vx * (t - tl), cy + vy * (t - tl)
            for ci, (x, y, _, _) in enumerate(cand):
                d = min(np.hypot(x - px, y - py), np.hypot(x - cx, y - cy))
                if d <= gate:
                    pairs.append((d, bi, ci))
        used_b, used_c = set(), set()
        for d, bi, ci in sorted(pairs):
            if bi in used_b or ci in used_c:
                continue
            used_b.add(bi)
            used_c.add(ci)
            b = blobs[bi]
            x, y, w, h = cand[ci]
            b.hist.append((t, x, y, w, h))
            b.miss, b.n = 0, b.n + 1
            if any(self._inside_frac(x, y, w, h, bx) >= 0.5 for _, bx in people + vehicles):
                b.known += 1
        for bi, b in enumerate(blobs):
            if bi not in used_b:
                b.miss += 1
                if b.miss > 2 or t - b.hist[-1][0] > 0.6:
                    self.blobs.pop(b.id, None)
        for ci, (x, y, w, h) in enumerate(cand):
            if ci in used_c:
                continue
            b = _Blob(self.next_id)
            self.next_id += 1
            b.hist.append((t, x, y, w, h))
            b.n = 1
            if any(self._inside_frac(x, y, w, h, bx) >= 0.5 for _, bx in people + vehicles):
                b.known = 1
            self.blobs[b.id] = b

        for b in list(self.blobs.values()):
            if b.miss:
                continue
            q = self._falling(b, t, c.min_samples)
            if q is None:
                q3 = self._falling(b, t, 3)            # đủ chặt để xét va trúng người
                if q3 is not None:
                    self._check_contact(b, q3, people, t)
                continue
            _, x, y, w, h = b.hist[-1]
            self.overlay.append({"box": self._to_raw([x - w / 2, y - h / 2, x + w / 2, y + h / 2], shift),
                                 "v": q["v"]})
            self._check_contact(b, q, people, t)
            if not b.reported and t - self.last_report >= c.cooldown:
                b.reported = True
                self.last_report = t
                events.append(self._object_event(b, q, people, t, shift))
        return events

    # ------------------------------------------------------------------
    def _falling(self, b: _Blob, t: float, min_samples: int) -> Optional[dict]:
        c = self.cfg
        pts = [p for p in b.hist if p[0] >= t - 1.2]
        if len(pts) < min_samples:
            return None
        ts = np.array([p[0] for p in pts])
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        H = float(self.size[1])
        dy, dx = (ys[-1] - ys[0]) / H, abs(xs[-1] - xs[0]) / H
        if dy < c.min_drop or dx > c.max_lateral * dy:
            return None
        if np.mean(np.diff(ys) > 0) < 0.8:
            return None
        if b.known / max(b.n, 1) >= 0.5:
            return None
        k = min(4, len(pts))
        v_end = float(np.polyfit(ts[-k:], ys[-k:], 1)[0]) / H
        if v_end < c.min_speed:
            return None
        steps = np.diff(ys) / np.maximum(np.diff(ts), 1e-3)   # tốc độ từng đoạn (px/s)
        half = max(1, len(steps) // 2)                         # phải tăng tốc: rơi tự do 1:3:5,
        if steps[-half:].mean() < 1.25 * max(steps[:half].mean(), 0.0):   # băng chuyền/cẩu hạ đều 1:1:1
            return None
        vx = float(np.polyfit(ts[-k:], xs[-k:], 1)[0])
        return {"v": v_end, "dy": dy, "vx": vx, "vy": v_end * H}

    def _object_event(self, b: _Blob, q: dict, people, t, shift) -> dict:
        _, x, y, w, h = b.hist[-1]
        box = self._to_raw([x - w / 2, y - h / 2, x + w / 2, y + h / 2], shift)
        near = None
        if people:
            ph = lambda bx: max(bx[3] - bx[1], 1.0)
            near = min(float(np.hypot(x - (bx[0] + bx[2]) / 2, y - (bx[1] + bx[3]) / 2)) / ph(bx)
                       for _, bx in people)
        conf = float(np.clip(0.5 + 0.25 * min(q["v"] / (3 * self.cfg.min_speed), 1.0)
                             + 0.15 * min(q["dy"] / 0.4, 1.0), 0.4, 0.9))
        d = {"speed_fh_s": round(q["v"], 2), "drop_fh": round(q["dy"], 2)}
        if near is not None:
            d["nearest_person_h"] = round(near, 1)
        return {"key": f"OBJECT_FALL:{b.id}:{t:.2f}", "type": "OBJECT_FALL", "severity": "warning",
                "confidence": round(conf, 2), "t": t, "track_ids": [f"O{b.id}"], "boxes": [box],
                "details": d}

    # ------------------------------------------------------------------
    def _check_contact(self, b: _Blob, q: dict, people, t: float) -> None:
        c = self.cfg
        _, x, y, w, h = b.hist[-1]
        dt = 0.1
        if len(b.hist) >= 2:
            dt = max(b.hist[-1][0] - b.hist[-2][0], 1e-3)
        nxt = (x + q["vx"] * dt, y + q["vy"] * dt)       # vị trí dự đoán ở khung sau
        for tr, bx in people:
            if tr.id in b.contacted:
                continue
            m = c.contact_margin * max(bx[3] - bx[1], 1.0)
            for px, py in ((x, y), nxt):
                if bx[0] - m <= px <= bx[2] + m and bx[1] - m <= py <= bx[3] + m:
                    b.contacted.add(tr.id)
                    self.contacts.append({"pid": tr.id, "t": t, "bid": b.id, "v": q["v"],
                                          "obj": [x - w / 2, y - h / 2, x + w / 2, y + h / 2]})
                    break

    @staticmethod
    def _resolve(tm: TrackManager, pid: str) -> Optional[Track]:
        tr = tm.tracks.get(pid)
        if tr is not None:
            return tr
        for p in tm.persons():                           # tracker đổi ID sau khi người ngã
            if p.state.get("inherited_from") == pid:
                return p
        return None

    def _contacts(self, tm: TrackManager, t: float, shift) -> List[dict]:
        c, out, keep = self.cfg, [], []
        for ct in self.contacts:
            if t - ct["t"] > c.confirm_window:
                continue
            tr = self._resolve(tm, ct["pid"])
            if tr is None:
                continue
            fs = tr.state.get("fall") or {}
            conf = reason = None
            if fs.get("phase") in FALLING_PHASES and fs.get("t_phase", -1e9) >= ct["t"] - 0.3 \
                    and t - ct["t"] > 0.1:
                conf, reason = 0.8, "Người ngã ngay sau khi bị vật rơi trúng"
            elif tr.lost and tr.lost_since is not None and t - tr.lost_since >= 0.3 \
                    and tr.lost_since >= ct["t"] - 0.1:
                conf, reason = 0.55, "Người mất dấu ngay sau khi bị vật rơi trúng"
            if conf is None:
                keep.append(ct)
                continue
            if t - self.last_struck.get(ct["pid"], -1e9) < c.cooldown * 2:
                continue
            self.last_struck[ct["pid"]] = t
            out.append({"key": f"STRUCK_BY:{ct['pid']}:{ct['bid']}:{t:.2f}", "type": "STRUCK_BY",
                        "severity": "critical", "confidence": conf, "t": t,
                        "track_ids": [ct["pid"], f"O{ct['bid']}"],
                        "boxes": [tr.last.box.tolist(), self._to_raw(ct["obj"], shift)],
                        "details": {"reason": reason, "object_speed_fh_s": round(ct["v"], 2)}})
        self.contacts = keep
        return out
