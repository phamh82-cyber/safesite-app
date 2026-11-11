"""Phát hiện ngã từ trên cao (giàn giáo, mái, thang, sàn tầng).

Khác ngã cùng mặt bằng (hông hạ tối đa ≈ 0,5 chiều cao người), người rơi từ trên cao có:
  • độ rơi lớn: hông hạ ≥ `min_drop` chiều cao người trong ≤ ~1 s,
  • rơi nhanh và tăng tốc (rơi tự do ≈ 5,8 H/s²; đi cầu thang, leo thang ≈ 0),
  • kích thước bbox gần như không đổi (người chạy lại gần camera thì bbox to lên – loại trừ).

IDLE ──rơi nhanh, sâu, tăng tốc──▶ FALLING ──người nằm (pha LYING/FALLEN của FallAnalyzer)──▶ sự kiện
                                       ├──mất dấu ≥ lost_confirm (kể cả rơi ra khỏi khung)──▶ sự kiện
                                       └──quá outcome_window mà vẫn đứng/đi lại──▶ IDLE (nhảy xuống bậc, hạ cánh an toàn)

Ước lượng độ cao rơi (est_drop_m) lấy chiều cao người 1,7 m làm chuẩn, chỉ mang tính tham khảo:
phối cảnh và góc nghiêng camera làm sai số.
Tọa độ ảnh đã trừ độ rung camera (Obs.shift).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np

from ..config import HeightFallSettings
from ..tracks import L_HIP, R_HIP, Track, TrackManager
from .fall import _mid

PERSON_H_M = 1.7
LYING_CONFIRM = 0.3       # giây: pha LYING phải kéo dài chút mới tính là đã nằm


class HeightFallAnalyzer:
    def __init__(self, cfg: HeightFallSettings):
        self.cfg = cfg
        self.recent: Dict[str, float] = {}      # track id -> thời điểm báo gần nhất

    # ------------------------------------------------------------------
    def recent_any(self, ids, t: float, within: float = 8.0) -> bool:
        return any(i in self.recent and t - self.recent[i] <= within for i in ids)

    @staticmethod
    def _y(tr: Track) -> float:
        o = tr.last
        k = o.kpts
        hp = _mid(k, L_HIP, R_HIP) if k is not None else None
        y = float(hp[1]) if hp is not None else 0.5 * float(o.box[1] + o.box[3])
        return y - float(o.shift[1])

    def _metrics(self, st: dict, t: float):
        """(v, accel, drop, growth, down, h_norm) hoặc None nếu chưa đủ mẫu.
        Đơn vị: chiều cao người, chuẩn hóa theo chiều cao bbox trung vị trong cửa sổ (không dùng h_ref
        vì h_ref có thể lỗi thời khi người đang tiến lại gần camera)."""
        c = self.cfg
        ts = np.array([x[0] for x in st["ys"]])
        if len(ts) < 4:
            return None
        dt = max(float(np.median(np.diff(ts))), 1e-3)
        sel = ts >= t - max(c.window, 4 * dt)          # FPS thấp: nới cửa sổ để đủ mẫu
        if sel.sum() < 4:
            return None
        tt = ts[sel]
        yy = np.array([x[1] for x in st["ys"]])[sel]
        hh = np.array([x[2] for x in st["ys"]])[sel]
        rec = tt >= t - max(0.4, 3.2 * dt)
        if rec.sum() < 3 or np.ptp(tt[rec]) < 1e-3:
            return None
        h_norm = max(float(np.median(hh)), 1.0)
        v = float(np.polyfit(tt[rec], yy[rec], 1)[0]) / h_norm
        accel = 2.0 * float(np.polyfit(tt - t, yy, 2)[0]) / h_norm
        drop = float(yy[-1] - yy.min()) / h_norm
        # bbox to lên ở BẤT KỲ thời điểm nào trong cửa sổ = đang tiến lại gần camera
        growth = float(np.max(hh / np.maximum(np.minimum.accumulate(hh), 1.0)))
        down = float(np.mean(np.diff(yy[rec]) > 0))      # chỉ xét đoạn gần nhất đang rơi
        return v, accel, drop, growth, down, h_norm

    # ------------------------------------------------------------------
    def step(self, tm: TrackManager, t: float, fall_events: List[dict]) -> List[dict]:
        c = self.cfg
        if not c.enabled:
            return []
        out: List[dict] = []
        self.recent = {k: v for k, v in self.recent.items() if t - v <= 30.0}
        for tr in tm.persons():
            st = tr.state.get("hfall")
            if tr.lost:
                if st and st["phase"] == "FALLING" and tr.lost_since is not None \
                        and t - tr.lost_since >= c.lost_confirm:
                    out.append(self._emit(tr, st, t, 0.6, "Mất dấu ngay khi đang rơi nhanh"))
                continue
            if tr.last_t != t:
                continue
            if st is None:
                st = tr.state["hfall"] = {"ys": deque(maxlen=60), "phase": "IDLE", "done_t": -1e9}
            o = tr.last
            h_px = max(float(o.box[3] - o.box[1]), 1.0)
            fs = tr.state.get("fall") or {}
            st["ys"].append((t, self._y(tr), h_px))
            m = self._metrics(st, t)

            if st["phase"] == "IDLE":
                if m is None or t - st["done_t"] < c.cooldown:
                    continue
                v, a, drop, growth, down, h_norm = m
                if v >= c.min_speed and drop >= c.min_drop and a >= c.min_accel \
                        and growth <= c.max_growth and down >= 0.66:
                    st.update(phase="FALLING", t0=t, h_norm=h_norm, peak=v, max_drop=drop,
                              y0=min(x[1] for x in st["ys"] if x[0] >= t - c.window))
            else:  # FALLING
                if m is not None:
                    st["peak"] = max(st["peak"], m[0])
                st["max_drop"] = max(st["max_drop"], (self._y(tr) - st["y0"]) / st["h_norm"])
                phase, t_ph = fs.get("phase"), fs.get("t_phase", t)
                if phase in ("LYING", "FALLEN") and t - t_ph >= (LYING_CONFIRM if phase == "LYING" else 0):
                    out.append(self._emit(tr, st, t, 0.85, "Rơi nhanh từ trên cao rồi nằm"))
                elif t - st["t0"] > c.outcome_window:
                    st["phase"] = "IDLE"          # hạ cánh an toàn / nhảy xuống bậc thấp
        return out

    def _emit(self, tr: Track, st: dict, t: float, conf: float, reason: str) -> dict:
        drop = float(st["max_drop"])
        # độ rơi càng sâu và càng nhanh thì càng đáng tin
        conf = float(np.clip(conf + 0.05 * min(drop - self.cfg.min_drop, 1.5), 0.3, 0.97))
        st.update(phase="IDLE", done_t=t)
        self.recent[tr.id] = t
        return {"key": f"FALL_FROM_HEIGHT:{tr.id}:{t:.2f}", "type": "FALL_FROM_HEIGHT",
                "severity": "critical", "confidence": round(conf, 2), "t": t,
                "track_ids": [tr.id], "boxes": [tr.last.box.tolist()],
                "details": {"reason": reason, "drop_h": round(drop, 2),
                            "est_drop_m": round(drop * PERSON_H_M, 1),
                            "fall_speed": round(float(st["peak"]), 2)}}
