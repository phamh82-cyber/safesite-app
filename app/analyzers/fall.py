"""Phát hiện té ngã dựa trên khung xương (pose) + máy trạng thái.

NORMAL ──rơi nhanh──▶ DESCENDING ──nằm──▶ LYING ──giữ lying_hold──▶ FALLEN ──bất động──▶ (FALL_IMMOBILE)
  │                        └──quá confirm_window không nằm──▶ NORMAL (chỉ là cúi/ngồi)
  └──nằm mà không có pha rơi──▶ LYING_SLOW ──bất động person_down_after──▶ DOWN (PERSON_DOWN)
FALLEN / DOWN ──đứng dậy recover_hold──▶ NORMAL
"""
from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from ..config import FallSettings
from ..tracks import L_HIP, L_SH, R_HIP, R_SH, Track, TrackManager

KP_CONF = 0.3


def _mid(k: np.ndarray, a: int, b: int):
    pts = [k[i, :2] for i in (a, b) if k[i, 2] >= KP_CONF]
    return np.mean(pts, 0) if pts else None


def frame_features(track: Track, h_ref: float) -> dict:
    o = track.last
    x1, y1, x2, y2 = o.box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    f = {"t": o.t, "ar": w / h, "h_ratio": h / h_ref, "valid": False, "theta": None,
         "hip_y": y1 + 0.55 * h, "kconf": 0.0, "box": (float(x1), float(y1), float(x2), float(y2))}
    k = o.kpts
    if k is not None:
        s, hp = _mid(k, L_SH, R_SH), _mid(k, L_HIP, R_HIP)
        f["kconf"] = float(np.mean(k[:, 2]))
        if s is not None and hp is not None:
            vec = s - hp
            n = float(np.linalg.norm(vec))
            if n > 2:
                f["theta"] = float(np.degrees(np.arccos(np.clip(-vec[1] / n, -1, 1))))
                f["valid"] = True
                f["hip_y"] = float(hp[1])
    # năng lượng chuyển động (chuẩn hóa theo H_ref, trên giây)
    f["energy"] = 0.0
    if len(track.hist) >= 2:
        p = track.hist[-2]
        dt = max(o.t - p.t, 1e-3)
        if o.kpts is not None and p.kpts is not None:
            m = (o.kpts[:, 2] >= KP_CONF) & (p.kpts[:, 2] >= KP_CONF)
            if m.sum() >= 4:
                d = np.linalg.norm(o.kpts[m, :2] - p.kpts[m, :2], axis=1)
                f["energy"] = float(np.mean(d)) / h_ref / dt
        if f["energy"] == 0.0:
            d = np.abs(np.asarray(o.box) - np.asarray(p.box)).mean()
            f["energy"] = float(d) / h_ref / dt
    return f


class FallAnalyzer:
    def __init__(self, cfg: FallSettings, img_w: int, img_h: int):
        self.cfg, self.w, self.h = cfg, img_w, img_h

    # ------------------------------------------------------------------
    def _touches_border(self, box, m: int = 4) -> bool:
        return box[1] <= m or box[3] >= self.h - m or box[0] <= m or box[2] >= self.w - m

    def _init_state(self, tr: Track) -> dict:
        x1, y1, x2, y2 = tr.last.box
        w, h = x2 - x1, y2 - y1
        return {"phase": "NORMAL", "heights": deque(maxlen=40),
                "h_ref": float(max(h, w) if w > 1.3 * h else h),
                "feat": deque(maxlen=120), "t_phase": tr.last_t}

    def _set(self, st: dict, phase: str, t: float) -> None:
        st["phase"], st["t_phase"] = phase, t

    def _lying(self, f: dict) -> bool:
        c = self.cfg
        if f["valid"] and f["theta"] >= c.lying_angle and f["h_ratio"] < 0.75:
            return True
        if f["ar"] >= c.lying_aspect:
            return True
        return f["h_ratio"] < 0.45 and (not f["valid"] or f["theta"] >= 40)

    def _upright(self, f: dict) -> bool:
        if f["valid"]:
            return f["theta"] < self.cfg.upright_angle and f["h_ratio"] > 0.75
        return f["ar"] < 0.6 and f["h_ratio"] > 0.8

    def _descent(self, st: dict, f: dict) -> tuple[bool, float]:
        c = self.cfg
        feats = [x for x in st["feat"] if x["t"] >= f["t"] - c.descent_window]
        if len(feats) < 3:
            return False, 0.0
        ts = np.array([x["t"] for x in feats])
        ys = np.array([x["hip_y"] for x in feats])
        recent = ts >= f["t"] - 0.4
        v = 0.0
        if recent.sum() >= 3 and np.ptp(ts[recent]) > 1e-3:
            v = float(np.polyfit(ts[recent], ys[recent], 1)[0]) / st["h_ref"]
        drop = max(x["h_ratio"] for x in feats) - f["h_ratio"]
        # thân đổ nhanh (góc tăng mạnh trong 0,5 s) – bắt được cú ngã trước khi detector mất người
        th = [x["theta"] for x in feats if x["theta"] is not None and x["t"] >= f["t"] - 0.5]
        tilt = f["valid"] and len(th) >= 2 and f["theta"] >= 35 and f["theta"] - min(th) >= c.torso_rate \
            and drop > 0.1
        return (v >= c.descent_speed or drop >= c.height_drop or tilt), v

    def _immobile_for(self, st: dict, t: float, dur: float) -> bool:
        """Bất động: vị trí và kích thước bbox gần như không đổi trong `dur` giây.
        Dùng bbox thay vì từng keypoint vì keypoint của người nằm rung nhiều."""
        feats = [x for x in st["feat"] if x["t"] >= t - dur]
        if len(feats) < 3 or feats[0]["t"] > t - dur + 0.5:
            return False
        b = np.array([x["box"] for x in feats])
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        spread = max(np.std(cx), np.std(cy), np.std(bw), np.std(bh)) / st["h_ref"]
        return spread <= self.cfg.immobile_spread

    def _conf(self, st: dict, f: dict) -> float:
        v = st.get("v_peak", 0.0) / (2 * self.cfg.descent_speed)
        pose = (f["theta"] / 90.0) if f["valid"] else min(f["ar"] / 2.0, 1.0)
        return float(np.clip(0.35 * min(v, 1) + 0.35 * min(pose, 1) + 0.3 * f["kconf"] + 0.1,
                             0.3, 0.99))

    # ------------------------------------------------------------------
    def step(self, tm: TrackManager, t: float) -> List[dict]:
        events: List[dict] = []
        c = self.cfg
        for tr in tm.persons():
            st = tr.state.get("fall")
            if tr.lost:
                # Mất dấu ngay sau pha rơi nhanh: detector thường không thấy người nằm
                if st and st["phase"] in ("DESCENDING", "LYING") and t - tr.lost_since >= 1.5 \
                        and not self._touches_border(tr.last.box) and not st.get("lost_alert"):
                    st["lost_alert"] = True
                    events.append(self._event(tr, "FALL", "critical", 0.5, t,
                                              {"reason": "Mất dấu ngay sau khi rơi nhanh"}))
                continue
            if tr.last_t != t:
                continue
            if st is None:
                st = tr.state["fall"] = self._init_state(tr)
            f = frame_features(tr, st["h_ref"])
            border = self._touches_border(tr.last.box)
            lying, upright = self._lying(f), self._upright(f)
            f["lying"], f["upright"] = lying, upright
            st["feat"].append(f)
            st["last_f"] = f
            ph = st["phase"]

            if ph == "NORMAL":
                if upright and not border:
                    st["heights"].append(tr.last.box[3] - tr.last.box[1])
                    st["h_ref"] = float(np.median(st["heights"]))
                desc, v = (False, 0.0) if border else self._descent(st, f)
                if desc:
                    st["v_peak"] = v
                    self._set(st, "DESCENDING", t)
                elif lying and not border:
                    self._set(st, "LYING_SLOW", t)
            elif ph == "DESCENDING":
                _, v = self._descent(st, f)
                st["v_peak"] = max(st.get("v_peak", 0.0), v)
                if lying:
                    st["t_desc"] = st["t_phase"]
                    self._set(st, "LYING", t)
                elif t - st["t_phase"] > c.confirm_window:
                    self._set(st, "NORMAL", t)
            elif ph == "LYING":
                if not lying:
                    st.setdefault("t_unlie", t)
                    if t - st["t_unlie"] > 0.5:
                        st.pop("t_unlie", None)
                        self._set(st, "NORMAL", t)
                else:
                    st.pop("t_unlie", None)
                    if t - st["t_phase"] >= c.lying_hold:
                        ev = self._event(tr, "FALL", "critical", self._conf(st, f), t, {
                            "torso_angle": None if f["theta"] is None else round(f["theta"], 1),
                            "fall_speed": round(st.get("v_peak", 0.0), 2),
                            "height_ratio": round(f["h_ratio"], 2)})
                        st["event_key"] = ev["key"]
                        events.append(ev)
                        self._set(st, "FALLEN", t)
            elif ph == "LYING_SLOW":
                if not lying:
                    self._set(st, "NORMAL", t)
                elif t - st["t_phase"] >= c.person_down_after and \
                        self._immobile_for(st, t, c.person_down_after):
                    ev = self._event(tr, "PERSON_DOWN", "warning", 0.6, t,
                                     {"reason": "Nằm bất động, không thấy pha rơi"})
                    st["event_key"] = ev["key"]
                    events.append(ev)
                    self._set(st, "DOWN", t)
            elif ph in ("FALLEN", "DOWN"):
                if upright:
                    st.setdefault("t_up", t)
                    if t - st["t_up"] >= c.recover_hold:
                        events.append({"update_of": st.get("event_key"),
                                       "details": {"recovered_after_s": round(t - st["t_phase"], 1)}})
                        st.pop("t_up", None)
                        st.pop("escalated", None)
                        self._set(st, "NORMAL", t)
                else:
                    st.pop("t_up", None)
                    if ph == "FALLEN" and not st.get("escalated") and \
                            t - st["t_phase"] >= c.immobile_after and \
                            self._immobile_for(st, t, c.immobile_after):
                        st["escalated"] = True
                        events.append(self._event(tr, "FALL_IMMOBILE", "critical", 0.8, t, {
                            "reason": f"Bất động > {c.immobile_after:.0f}s sau khi ngã"}))
        return events

    @staticmethod
    def _event(tr: Track, typ: str, sev: str, conf: float, t: float, details: dict) -> dict:
        return {"key": f"{typ}:{tr.id}:{t:.2f}", "type": typ, "severity": sev,
                "confidence": round(float(conf), 2), "t": t, "track_ids": [tr.id],
                "boxes": [tr.last.box.tolist()], "details": details}


def inherit_fall_state(tm: TrackManager, new_tr: Track) -> None:
    """Người vừa ngã thường bị tracker cấp ID mới (detector mất người ở tư thế nằm).
    Track mới xuất hiện gần một track vừa mất dấu thì nhận lại trạng thái của nó:
    - track cũ đang ở pha ngã  → giữ nguyên pha;
    - track cũ đang đứng nhưng track mới đã ở tư thế nằm → coi như đang rơi
      (DESCENDING tính từ lúc track cũ mất dấu)."""
    o = new_tr.last
    size = max(o.box[3] - o.box[1], o.box[2] - o.box[0])
    old = tm.find_lost_person_near(o.foot_img, o.t, radius_px=max(80.0, 1.2 * size))
    if old is None:
        old = tm.find_lost_person_near(o.foot_img, o.t, radius_px=max(80.0, 1.2 * size),
                                       max_age=2.0, allow_normal=True)
        if old is None:
            return
        st = old.state.get("fall")
        h_ref = st["h_ref"]
        w, h = o.box[2] - o.box[0], o.box[3] - o.box[1]
        if not (w / max(h, 1) >= 1.0 or h / h_ref < 0.6):
            return
        st["phase"], st["t_phase"] = "DESCENDING", old.lost_since
        st["id_switch"] = True
    new_tr.state["fall"] = old.state.pop("fall")
    if "hfall" in old.state:                   # giữ cả pha rơi từ trên cao (height_fall.py)
        new_tr.state["hfall"] = old.state.pop("hfall")
    new_tr.state["inherited_from"] = old.id
    tm.tracks.pop(old.id, None)
