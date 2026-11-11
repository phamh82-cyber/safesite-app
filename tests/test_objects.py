"""Kiểm thử vật rơi (OBJECT_FALL) và bị vật đập trúng (STRUCK_BY) bằng video tổng hợp:
nền có vân + nhiễu cảm biến, vật là hình chữ nhật tối rơi theo gia tốc."""
from __future__ import annotations

import cv2
import numpy as np

from tests.sim import STAND, person_det, posture_points
from tests.test_scenarios import H_IMG, W, make_engine, types

FPS = 10.0
DT = 1.0 / FPS
HP, FX, FY = 300, 640, 600            # người đứng: chân (640, 600)


def make_bg(seed=0):
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur(rng.integers(0, 255, (H_IMG, W), dtype=np.uint8), (0, 0), 3)
    return cv2.normalize(g, None, 50, 205, cv2.NORM_MINMAX).astype(np.float32)


def render(bg, rng, rects=(), shift=(0, 0), gain=1.0):
    img = bg.copy()
    for (x1, y1, x2, y2, col) in rects:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), float(col), -1)
    if shift != (0, 0):
        m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        img = cv2.warpAffine(img, m, (W, H_IMG), borderMode=cv2.BORDER_REFLECT)
    img = np.clip(img * gain + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def obj_rect(cx, cy, s=40):
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, 25)


def run_video(n_frames, rects_at, dets_at=lambda i: [], shift_at=lambda i: (0, 0), gain_at=lambda i: 1.0,
              eng=None, seed=0):
    eng, rng, bg, out = eng or make_engine(), np.random.default_rng(seed + 1), make_bg(seed), []
    for i in range(n_frames):
        frame = render(bg, rng, rects_at(i), shift_at(i), gain_at(i))
        out += [e for e in eng.process(i * DT, dets_at(i), frame) if "type" in e]
    return out


WARM = 45                                  # 4,5 s nền tĩnh cho MOG2 học


def fall_y(i, i0, g=1500.0, y0=0.0):       # vật bắt đầu rơi từ khung i0
    t = (i - i0) * DT
    return y0 + 0.5 * g * t * t


def test_falling_object_detected_once():
    n = WARM + 14
    ev = run_video(n, lambda i: [obj_rect(900, fall_y(i, WARM))] if i >= WARM else [])
    assert types(ev) == ["OBJECT_FALL"], types(ev)


def test_bird_moving_sideways_no_alert():
    ev = run_video(WARM + 20, lambda i: [obj_rect(100 + 70 * (i - WARM), 200 + 25 * np.sin(i))]
                   if i >= WARM else [])
    assert not ev, types(ev)


def test_slow_drifting_leaf_no_alert():
    ev = run_video(WARM + 30, lambda i: [obj_rect(700, 100 + 40 * (i - WARM) * DT)] if i >= WARM else [])
    assert not ev, types(ev)


def test_constant_speed_descent_not_free_fall():
    """Vật tụt đều 0,55 chiều cao khung/giây (băng chuyền, vật bị cẩu hạ): không tăng tốc ⇒ không báo."""
    ev = run_video(WARM + 14, lambda i: [obj_rect(700, 60 + 400 * (i - WARM) * DT)] if i >= WARM else [])
    assert not ev, types(ev)


def test_rain_of_small_streaks_no_alert():
    rng = np.random.default_rng(5)
    xs = rng.integers(50, W - 50, 60)
    ph = rng.random(60)

    def rects(i):
        if i < WARM:
            return []
        return [(x, ((i * 90 + p * 700) % H_IMG), x + 3, ((i * 90 + p * 700) % H_IMG) + 14, 40)
                for x, p in zip(xs, ph)]
    assert not run_video(WARM + 25, rects)


def test_global_brightness_jump_no_alert():
    ev = run_video(WARM + 12, lambda i: [], gain_at=lambda i: 1.0 if i < WARM + 3 else 1.6)
    assert not ev, types(ev)


def person_dets(n_topple_from=None, fx=FX):
    rng = np.random.default_rng(3)

    def dets(i):
        j = 0 if n_topple_from is None or i < n_topple_from else i - n_topple_from
        k = posture_points(STAND, fx, FY, HP, min(90, 12 * j), 0.004, rng)
        return [person_det(1, k, HP)]
    return dets


def test_object_hits_person_who_falls():
    """Vật rơi trúng đầu người, 0,3 s sau người đổ: OBJECT_FALL + STRUCK_BY (+ FALL thường)."""
    hit = WARM + 7
    n = WARM + 40
    ev = run_video(n, lambda i: [obj_rect(FX, fall_y(i, WARM))] if WARM <= i <= hit else [],
                   person_dets(hit + 3))
    t = types(ev)
    assert "OBJECT_FALL" in t and t.count("STRUCK_BY") == 1 and "FALL" in t, t


def test_object_falls_beside_person_no_struck():
    hit = WARM + 7
    ev = run_video(WARM + 40, lambda i: [obj_rect(FX + 350, fall_y(i, WARM))] if WARM <= i <= hit else [],
                   person_dets(hit + 3))
    t = types(ev)
    assert "STRUCK_BY" not in t, t


def test_hit_without_reaction_is_warning_only():
    """Giới hạn đã biết: trúng nhưng người vẫn đứng ⇒ chỉ cảnh báo vật rơi, không có STRUCK_BY."""
    hit = WARM + 7
    ev = run_video(WARM + 40, lambda i: [obj_rect(FX, fall_y(i, WARM))] if WARM <= i <= hit else [],
                   person_dets(None))
    t = types(ev)
    assert "STRUCK_BY" not in t and "OBJECT_FALL" in t, t


def test_person_falling_alone_is_not_an_object():
    """Bóng người đổ xuống (vẽ thành khối tối) không được coi là vật rơi."""
    t0 = WARM + 2
    dets = person_dets(t0)

    def rects(i):
        d = dets(i)[0]
        return [(*d.box, 30)] if i >= WARM else []
    ev = run_video(WARM + 30, rects, dets)
    assert "OBJECT_FALL" not in types(ev) and "STRUCK_BY" not in types(ev), types(ev)


def test_accelerating_but_mostly_sideways_no_alert():
    """Vật tăng tốc đi xuống nhưng bay ngang nhiều hơn rơi (chim lao, mảnh bay): chỉ tiêu chí
    'chủ yếu theo phương đứng' loại được ca này."""
    ev = run_video(WARM + 14, lambda i: [obj_rect(100 + 900 * (i - WARM) * DT,
                                                  fall_y(i, WARM, g=700, y0=150))] if i >= WARM else [])
    assert not ev, types(ev)


def test_slow_accelerating_below_speed_floor_no_alert():
    """Tăng tốc rất chậm (g nhỏ): rơi đủ xa nhưng tốc độ cuối < min_speed ⇒ không báo."""
    ev = run_video(WARM + 40, lambda i: [obj_rect(700, fall_y(i, WARM, g=50, y0=100))] if i >= WARM else [])
    assert not ev, types(ev)
