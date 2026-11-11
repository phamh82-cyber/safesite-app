"""Phát hiện nguy cơ va chạm và va chạm: người – phương tiện, phương tiện – phương tiện.
Mọi phép đo làm trên mặt phẳng nền (mét), không dùng chồng lấn bbox trên ảnh làm
tiêu chí chính – bbox chồng nhau có thể chỉ do người đứng phía sau xe."""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from ..config import CollisionSettings
from ..geometry import box_iou_any, closest_point_on_polygon, point_polygon_distance, polygon_distance
from ..tracks import Track, TrackManager

FALLING_PHASES = ("DESCENDING", "LYING", "FALLEN")


def _heading(v: np.ndarray) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _ang_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


class CollisionAnalyzer:
    def __init__(self, cfg: CollisionSettings, img_w: int, img_h: int):
        self.cfg, self.w, self.h = cfg, img_w, img_h
        self.pairs: Dict[Tuple[str, str], dict] = {}
        self.overlay: List[dict] = []   # để vẽ lên khung hình

    def _pair(self, a: str, b: str) -> dict:
        return self.pairs.setdefault((a, b), {"near_since": None, "last_alert": {},
                                              "contact_t": None, "vc_hist": deque(maxlen=60)})

    def _cool(self, ps: dict, typ: str, t: float) -> bool:
        last = ps["last_alert"].get(typ)
        return last is None or t - last >= self.cfg.cooldown

    def _border(self, box, m: int = 4) -> bool:
        return box[1] <= m or box[3] >= self.h - m or box[0] <= m or box[2] >= self.w - m

    # ------------------------------------------------------------------
    def step(self, tm: TrackManager, t: float, fall_events: List[dict]) -> List[dict]:
        self.overlay = []
        events: List[dict] = []
        persons = tm.persons()
        vehicles = [v for v in tm.vehicles() if not v.lost and v.last.footprint is not None]
        for p in persons:
            for v in vehicles:
                events += self._person_vehicle(p, v, t)
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                events += self._vehicle_vehicle(vehicles[i], vehicles[j], t)
        events += self._fuse_falls(fall_events, t)
        alive = set(tm.tracks)
        for k in [k for k in self.pairs if k[0] not in alive or k[1] not in alive]:
            del self.pairs[k]
        return events

    # ------------------------------------------------------------------
    def _person_vehicle(self, p: Track, v: Track, t: float) -> List[dict]:
        c, out = self.cfg, []
        ps = self._pair(p.id, v.id)
        if p.lost:
            # chỉ quan tâm người mất dấu khi đang chờ xác nhận va chạm
            if ps["contact_t"] is not None and t - ps["contact_t"] <= c.confirm_window \
                    and not self._border(p.last.box) and self._cool(ps, "PV", t):
                out.append(self._ev("PERSON_VEHICLE_COLLISION", p, v, t, 0.6, ps,
                                    {"reason": "Người mất dấu ngay sau khi tiếp xúc xe"}))
                ps["contact_t"] = None
            return out
        pg, poly = p.last.ground, v.last.footprint
        d = point_polygon_distance(pg, poly)
        cp = closest_point_on_polygon(pg, poly)
        rel = pg - (cp if d > 1e-6 else poly.mean(0))
        n = float(np.linalg.norm(rel)) or 1e-6
        u = rel / n
        v_rel = p.velocity() - v.velocity()
        vc = -float(np.dot(rel, v_rel)) / n
        veh_toward = float(np.dot(v.velocity(), u))      # thành phần vận tốc xe hướng về người
        vs = v.speed()
        ttc = d / vc if vc > 0.3 else math.inf
        if d < 0.6 * c.near_distance + 0.4 and (vs > 0.5 or v.max_speed_since(t - 1.5) > 0.5):
            ps["last_close_t"] = t

        near = (d < c.near_distance and vs > c.min_vehicle_speed) or \
               (ttc < c.near_ttc and veh_toward > 0.3 and d < 3 * c.near_distance)
        level = "danger" if near else ("watch" if d < 2 * c.near_distance else None)
        if level:
            self.overlay.append({"a": p.id, "b": v.id, "pa": p.last.foot_img, "pb": v.last.box,
                                 "d": d, "ttc": ttc, "level": level})
        if near:
            ps["near_since"] = ps["near_since"] or t
            if t - ps["near_since"] >= c.near_hold and self._cool(ps, "NEAR", t):
                conf = 0.5 + 0.25 * max(0.0, 1 - d / c.near_distance) + \
                       0.24 * (0 if math.isinf(ttc) else max(0.0, 1 - ttc / c.near_ttc))
                out.append(self._ev("NEAR_MISS", p, v, t, conf, ps, {
                    "distance_m": round(d, 2), "ttc_s": None if math.isinf(ttc) else round(ttc, 2),
                    "vehicle_speed_ms": round(vs, 2)}, sev="warning", cool_key="NEAR"))
        else:
            ps["near_since"] = None

        # ---- tiếp xúc + xác nhận ----
        moving = vs > 0.3 or v.max_speed_since(t - 1.0) > 0.5 or vc > 0.5
        if ps["contact_t"] is None and d < c.contact_distance and \
                box_iou_any(p.last.box, v.last.box) and moving:
            ps["contact_t"] = t
            ps["v_pre"] = p.velocity().copy()
            ps["contact_d"] = d
        if ps["contact_t"] is not None:
            dt = t - ps["contact_t"]
            fall_phase = (p.state.get("fall") or {}).get("phase")
            dv = float(np.linalg.norm(p.velocity() - ps["v_pre"]))
            pushed = dt > 0.1 and dv > 1.0 and dv / dt > c.push_accel
            if (fall_phase in FALLING_PHASES or pushed) and self._cool(ps, "PV", t):
                reason = "Người ngã sau khi tiếp xúc xe" if fall_phase in FALLING_PHASES \
                    else "Người bị đẩy đột ngột khi tiếp xúc xe"
                out.append(self._ev("PERSON_VEHICLE_COLLISION", p, v, t, 0.85, ps, {
                    "reason": reason, "distance_m": round(ps["contact_d"], 2),
                    "vehicle_speed_ms": round(vs, 2)}))
                ps["contact_t"] = None
            elif dt > c.confirm_window:
                ps["contact_t"] = None
                if self._cool(ps, "NEAR", t):
                    out.append(self._ev("NEAR_MISS", p, v, t, 0.7, ps, {
                        "reason": "Tiếp xúc rất gần xe đang chạy, chưa thấy dấu hiệu bị thương",
                        "distance_m": round(ps["contact_d"], 2)}, sev="warning", cool_key="NEAR"))
        return out

    # ------------------------------------------------------------------
    def _vehicle_vehicle(self, a: Track, b: Track, t: float) -> List[dict]:
        c, out = self.cfg, []
        ps = self._pair(a.id, b.id)
        fa, fb = a.last.footprint, b.last.footprint
        d = polygon_distance(fa, fb)
        rel = fa.mean(0) - fb.mean(0)
        n = float(np.linalg.norm(rel)) or 1e-6
        vc = -float(np.dot(rel, a.velocity() - b.velocity())) / n
        ps["vc_hist"].append((t, vc))
        if d < 1.5:
            self.overlay.append({"a": a.id, "b": b.id, "pa": a.last.box, "pb": b.last.box,
                                 "d": d, "ttc": math.inf, "level": "watch", "vv": True})
        max_vc = max((x for tt, x in ps["vc_hist"] if tt >= t - 1.5), default=0.0)
        if ps["contact_t"] is None and d < c.vv_contact_distance and \
                box_iou_any(a.last.box, b.last.box) and max_vc > c.vv_min_closing:
            ps["contact_t"] = t
            ps["pre"] = {tr.id: (tr.max_speed_since(t - 1.0), _heading(tr.velocity()))
                         for tr in (a, b)}
            ps["closing"] = max_vc
        if ps["contact_t"] is not None:
            dt = t - ps["contact_t"]
            hit, why = False, ""
            for tr in (a, b):
                s0, h0 = ps["pre"][tr.id]
                s1 = tr.speed()
                if s0 > 1.0 and s1 < (1 - c.vv_speed_drop) * s0:
                    hit, why = True, f"{tr.id} giảm tốc {s0:.1f}→{s1:.1f} m/s"
                elif s0 > 1.0 and s1 > 0.5 and _ang_diff(h0, _heading(tr.velocity())) > c.vv_heading_change:
                    hit, why = True, f"{tr.id} đổi hướng đột ngột"
            if hit and dt > 0.1 and self._cool(ps, "VV", t):
                out.append(self._ev("VEHICLE_COLLISION", a, b, t, 0.8, ps, {
                    "reason": why, "closing_speed_ms": round(ps["closing"], 2)}, cool_key="VV"))
                ps["contact_t"] = None
            elif dt > 1.0:
                ps["contact_t"] = None
        return out

    # ------------------------------------------------------------------
    def _fuse_falls(self, fall_events: List[dict], t: float) -> List[dict]:
        """Một FALL xảy ra ≤ 3 s sau khi người ở rất gần xe đang chạy ⇒ va chạm người – xe."""
        out = []
        for ev in fall_events:
            if ev.get("type") != "FALL":
                continue
            pid = ev["track_ids"][0]
            for (a, b), ps in self.pairs.items():
                if a == pid and ps.get("last_close_t") and t - ps["last_close_t"] <= 3.0 \
                        and self._cool(ps, "PV", t):
                    ps["last_alert"]["PV"] = t
                    out.append({"key": f"PERSON_VEHICLE_COLLISION:{a}:{b}:{t:.2f}",
                                "type": "PERSON_VEHICLE_COLLISION", "severity": "critical",
                                "confidence": 0.75, "t": t, "track_ids": [a, b],
                                "boxes": ev["boxes"],
                                "details": {"reason": "Té ngã ngay sau khi ở sát xe đang chạy",
                                            "related": ev["key"]}})
        return out

    def _ev(self, typ, a: Track, b: Track, t, conf, ps, details, sev="critical", cool_key="PV"):
        ps["last_alert"][cool_key] = t
        return {"key": f"{typ}:{a.id}:{b.id}:{t:.2f}", "type": typ, "severity": sev,
                "confidence": round(float(min(conf, 0.99)), 2), "t": t,
                "track_ids": [a.id, b.id], "boxes": [a.last.box.tolist(), b.last.box.tolist()],
                "details": details}
