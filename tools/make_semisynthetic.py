"""Tạo video kiểm thử bán giả lập: cắt một người thật đang đứng từ video, ghép lên
nền trống rồi xoay quanh bàn chân để mô phỏng cú ngã (không thay thế dữ liệu thật).

    python -m tools.make_semisynthetic SRC.mp4 --frame 50 --box 472,80,556,260 --bg 100 --out fall_synth.mp4
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def paste_rotated(bg, crop, foot, angle):
    h, w = crop.shape[:2]
    big = int(np.hypot(h, w)) * 2 + 4
    canvas = np.zeros((big, big, 3), np.uint8)
    mask = np.zeros((big, big), np.uint8)
    cx, cy = big // 2, big // 2
    x0, y0 = cx - w // 2, cy - h          # bàn chân tại tâm canvas
    canvas[y0:y0 + h, x0:x0 + w] = crop
    mask[y0:y0 + h, x0:x0 + w] = 255
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    canvas = cv2.warpAffine(canvas, M, (big, big))
    mask = cv2.warpAffine(mask, M, (big, big))
    out = bg.copy()
    fx, fy = foot
    X0, Y0 = fx - cx, fy - cy
    xs, ys = max(0, X0), max(0, Y0)
    xe, ye = min(bg.shape[1], X0 + big), min(bg.shape[0], Y0 + big)
    sub_m = mask[ys - Y0:ye - Y0, xs - X0:xe - X0, None] / 255.0
    out[ys:ye, xs:xe] = (out[ys:ye, xs:xe] * (1 - sub_m) + canvas[ys - Y0:ye - Y0, xs - X0:xe - X0] * sub_m).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--bg", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="fall", choices=["fall", "walk"])
    ap.add_argument("--fps", type=float, default=12.0)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.src)
    cap.set(1, a.frame); _, f = cap.read()
    cap.set(1, a.bg); _, bg = cap.read()
    x1, y1, x2, y2 = map(int, a.box.split(","))
    pad = 6
    crop = f[y1 - pad:y2, x1 - pad:x2 + pad]
    foot = ((x1 + x2) // 2, y2)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (bg.shape[1], bg.shape[0]))
    rng = np.random.default_rng(0)
    n = int(9 * a.fps)
    for i in range(n):
        t = i / a.fps
        if a.mode == "fall":
            ang = 0 if t < 2 else min(85.0, 85.0 * ((t - 2) / 0.7) ** 1.6)
            fp = foot
        else:
            ang, fp = 0, (int(foot[0] - 25 * t), foot[1] + int(4 * t))
        jit = rng.integers(-1, 2, 2)
        img = paste_rotated(bg, crop, (fp[0] + jit[0], fp[1] + jit[1]), ang)
        wr.write(img)
    wr.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
