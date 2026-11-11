"""Vẽ kết quả lên khung hình (chữ ASCII vì OpenCV không hiển thị dấu tiếng Việt)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .engine import Engine
from .tracks import SKELETON

PHASE_COLOR = {"NORMAL": (80, 200, 80), "DESCENDING": (0, 220, 255), "LYING": (0, 140, 255),
               "LYING_SLOW": (0, 140, 255), "FALLEN": (0, 0, 255), "DOWN": (0, 0, 255)}
EV_TEXT = {"FALL": "FALL DETECTED", "FALL_IMMOBILE": "IMMOBILE AFTER FALL",
           "PERSON_DOWN": "PERSON DOWN", "NEAR_MISS": "NEAR MISS",
           "PERSON_VEHICLE_COLLISION": "PERSON-VEHICLE COLLISION",
           "VEHICLE_COLLISION": "VEHICLE COLLISION", "FALL_FROM_HEIGHT": "FALL FROM HEIGHT",
           "OBJECT_FALL": "FALLING OBJECT", "STRUCK_BY": "STRUCK BY OBJECT", "FIRE_SMOKE": "FIRE / SMOKE"}
VEH_COLOR = (255, 160, 40)
OBJ_COLOR = (255, 0, 200)
HAZ_COLOR = {"fire": (0, 90, 255), "smoke": (150, 150, 150)}


def _label(img, text, org, color, scale=0.5):
    k = max(1.0, img.shape[1] / 1280)
    scale, th = scale * k, max(1, int(round(k)))
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th,
                cv2.LINE_AA)


def draw(frame: np.ndarray, eng: Engine, recent_events: list, show_bev: bool = True) -> np.ndarray:
    img = frame.copy()
    lw = max(2, int(round(img.shape[1] / 640)))
    for tr in eng.tm.tracks.values():
        if tr.lost:
            continue
        o = tr.last
        x1, y1, x2, y2 = map(int, o.box)
        if tr.kind == "person":
            phase = (tr.state.get("fall") or {}).get("phase", "NORMAL")
            col = PHASE_COLOR.get(phase, (80, 200, 80))
            if o.kpts is not None:
                k = o.kpts
                for a, b in SKELETON:
                    if k[a, 2] > 0.3 and k[b, 2] > 0.3:
                        cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), col, lw,
                                 cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, lw)
            _label(img, f"{tr.id} {phase}", (x1, y1), col)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), VEH_COLOR, lw)
            _label(img, f"{tr.id} {tr.cls_name} {tr.speed():.1f}m/s", (x1, y1), VEH_COLOR)
    for ob in eng.objects.overlay:                      # vật đang rơi
        x1, y1, x2, y2 = map(int, ob["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), OBJ_COLOR, lw)
        _label(img, f"OBJ {ob['v']:.1f}fh/s", (x1, y1), OBJ_COLOR, 0.45)
    for kind, conf, box in eng.hazard.last_boxes:      # cháy / khói
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), HAZ_COLOR.get(kind, (0, 0, 255)), lw)
        _label(img, f"{kind} {conf:.2f}", (x1, y1), HAZ_COLOR.get(kind, (0, 0, 255)), 0.5)
    for ov in eng.collision.overlay:
        col = (0, 0, 255) if ov["level"] == "danger" else (0, 200, 255)
        if ov.get("vv"):
            a, b = ov["pa"], ov["pb"]
            pa = ((a[0] + a[2]) / 2, a[3])
        else:
            b = ov["pb"]
            pa = ov["pa"]
        pb = ((b[0] + b[2]) / 2, b[3])
        pa, pb = tuple(map(int, pa)), tuple(map(int, pb))
        cv2.line(img, pa, pb, col, lw, cv2.LINE_AA)
        txt = f"{ov['d']:.1f}m" + ("" if math.isinf(ov["ttc"]) else f" TTC {ov['ttc']:.1f}s")
        _label(img, txt, ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2), col, 0.45)
    if recent_events:
        h, w = img.shape[:2]
        k = max(1.0, w / 1280)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(44 * k)), (0, 0, 200), -1)
        img = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
        text = " | ".join(dict.fromkeys(EV_TEXT.get(e, e) for e in recent_events))
        cv2.putText(img, "! " + text, (int(12 * k), int(30 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k,
                    (255, 255, 255), max(2, int(2 * k)), cv2.LINE_AA)
    if show_bev:
        _bev(img, eng)
    return img


def _bev(img: np.ndarray, eng: Engine) -> None:
    """Bản đồ nhìn từ trên xuống (mặt nền) ở góc phải dưới."""
    h, w = img.shape[:2]
    size = int(max(120, min(h, w) * 0.26))
    pts, polys = [], []
    for tr in eng.tm.tracks.values():
        if tr.lost:
            continue
        if tr.kind == "person":
            pts.append((tr.last.ground, PHASE_COLOR.get((tr.state.get("fall") or {}).get("phase"),
                                                       (80, 200, 80))))
        elif tr.last.footprint is not None:
            polys.append(tr.last.footprint)
    allp = [p for p, _ in pts] + [q for poly in polys for q in poly]
    x0, y0 = w - size - 10, h - size - 10
    panel = img[y0:y0 + size, x0:x0 + size]
    panel[:] = (panel * 0.25 + np.array([30, 30, 30]) * 0.75).astype(np.uint8)
    cv2.putText(img, f"ground: {eng.mapper.mode}", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (230, 230, 230), 1, cv2.LINE_AA)
    if not allp:
        return
    arr = np.array(allp)
    lo, hi = arr.min(0) - 2, arr.max(0) + 2
    span = max(float((hi - lo).max()), 8.0)
    c = (lo + hi) / 2
    s = (size - 30) / span

    def P(p):
        return int(x0 + size / 2 + (p[0] - c[0]) * s), int(y0 + 18 + (size - 24) / 2 + (p[1] - c[1]) * s)

    for poly in polys:
        cv2.polylines(img, [np.array([P(q) for q in poly], np.int32)], True, VEH_COLOR, 2)
    for p, col in pts:
        cv2.circle(img, P(p), 4, col, -1)
    # thước tỉ lệ 2 m
    bx = x0 + 10
    cv2.line(img, (bx, y0 + size - 8), (bx + int(2 * s), y0 + size - 8), (230, 230, 230), 1)
    cv2.putText(img, "2m", (bx, y0 + size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
