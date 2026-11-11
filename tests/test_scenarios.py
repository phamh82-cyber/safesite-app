"""Kiểm thử logic phân tích bằng kịch bản giả lập.
Chạy:  python -m pytest -q tests   hoặc   python -m tests.test_scenarios"""
from __future__ import annotations

import numpy as np

from app.config import Settings
from app.engine import Engine
from tests.sim import KNEEL, SQUAT, STAND, blend, person_det, posture_points, vehicle_det

W, H_IMG, FPS = 1280, 720, 10.0
DT = 1.0 / FPS


def make_engine(calibrated: bool = False) -> Engine:
    eng = Engine(Settings(), W, H_IMG)
    if calibrated:  # 1 px = 1 cm trên nền
        eng.set_calibration([[0, 0], [W, 0], [W, H_IMG], [0, H_IMG]],
                            [[0, 0], [12.8, 0], [12.8, 7.2], [0, 7.2]])
    return eng


def run(eng: Engine, frames):
    events = []
    for i, dets in enumerate(frames):
        events += [e for e in eng.process(i * DT, dets) if "type" in e]
    return events


def types(evs):
    return [e["type"] for e in evs]


def person_seq(segments, fx=640, fy=600, Hp=300, seed=0, tid=1):
    """segments: list (duration_s, fn(s)->(posture, topple)). Trả về list detection mỗi frame."""
    rng = np.random.default_rng(seed)
    out = []
    for dur, fn in segments:
        n = int(round(dur * FPS))
        for j in range(n):
            post, topple = fn(j / max(n - 1, 1))
            k = posture_points(post, fx, fy, Hp, topple, jitter=0.004, rng=rng)
            out.append([person_det(tid, k, Hp)])
    return out


STANDING = (2.0, lambda s: (STAND, 0))


# ---------------------------------------------------------------- té ngã
def test_fall_detected():
    frames = person_seq([STANDING, (0.8, lambda s: (STAND, 90 * s ** 1.5)),
                         (4.0, lambda s: (STAND, 90))])
    evs = run(make_engine(), frames)
    falls = [e for e in evs if e["type"] == "FALL"]
    assert len(falls) == 1, types(evs)
    assert 2.8 <= falls[0]["t"] <= 4.5, falls[0]["t"]


def test_kneel_not_fall():
    frames = person_seq([STANDING, (1.0, lambda s: (blend(STAND, KNEEL, s), 0)),
                         (6.0, lambda s: (KNEEL, 3 * np.sin(12 * s))), STANDING])
    assert not run(make_engine(), frames), "Quỳ gối bị báo nhầm"


def test_fast_squat_not_fall():
    frames = person_seq([STANDING, (0.4, lambda s: (blend(STAND, SQUAT, s), 0)),
                         (2.0, lambda s: (SQUAT, 0)), (0.5, lambda s: (blend(SQUAT, STAND, s), 0)),
                         STANDING])
    assert not run(make_engine(), frames), "Ngồi xổm nhanh bị báo nhầm"


def test_slow_lie_down_person_down():
    frames = person_seq([STANDING, (4.0, lambda s: (STAND, 90 * s)), (8.0, lambda s: (STAND, 90))])
    t = types(run(make_engine(), frames))
    assert "FALL" not in t and "PERSON_DOWN" in t, t


def test_fall_then_immobile_escalates():
    frames = person_seq([STANDING, (0.8, lambda s: (STAND, 90 * s ** 1.5)),
                         (13.0, lambda s: (STAND, 90))])
    t = types(run(make_engine(), frames))
    assert t.count("FALL") == 1 and "FALL_IMMOBILE" in t, t


def test_fall_and_recover_no_escalation():
    frames = person_seq([STANDING, (0.8, lambda s: (STAND, 90 * s ** 1.5)),
                         (3.0, lambda s: (STAND, 90)), (1.5, lambda s: (STAND, 90 * (1 - s))),
                         (12.0, lambda s: (STAND, 0))])
    t = types(run(make_engine(), frames))
    assert t == ["FALL"], t


def test_lost_after_fast_descent():
    frames = person_seq([STANDING, (0.5, lambda s: (STAND, 85 * s))]) + [[] for _ in range(25)]
    t = types(run(make_engine(), frames))
    assert t == ["FALL"], t


def test_id_switch_keeps_fall_state():
    a = person_seq([STANDING, (0.8, lambda s: (STAND, 90 * s ** 1.5))], tid=1)
    b = person_seq([(4.0, lambda s: (STAND, 90))], tid=7)   # tracker đổi ID khi nằm
    t = types(run(make_engine(), a + b))
    assert t == ["FALL"], t


def test_id_switch_before_descent_detected():
    """Detector mất người ngay đầu cú ngã, tracker cấp ID mới khi người đã nằm."""
    a = person_seq([STANDING, (0.3, lambda s: (STAND, 20 * s))], tid=1)
    gap = [[] for _ in range(3)]
    b = person_seq([(4.0, lambda s: (STAND, 88))], tid=9)
    t = types(run(make_engine(), a + gap + b))
    assert t == ["FALL"], t


def test_walk_past_and_new_person_standing_no_fall():
    """Người A rời khung hình, người B đứng gần đó xuất hiện: không được nối nhầm."""
    a = person_seq([STANDING], tid=1)
    b = person_seq([(4.0, lambda s: (STAND, 0))], fx=660, tid=2)
    assert not run(make_engine(), a + [[]] + b)


# ---------------------------------------------------------------- va chạm
def static_person(n, fx=640, fy=600, tid=1, seed=1):
    rng = np.random.default_rng(seed)
    return [person_det(tid, posture_points(STAND, fx, fy, 300, 0, 0.004, rng), 300) for _ in range(n)]


def test_vehicle_passes_close_near_miss_only():
    n = 40
    ps = static_person(n)
    frames = [[ps[i], vehicle_det(1, 100 + 30 * i, 500, 200, 120)] for i in range(n)]
    t = types(run(make_engine(True), frames))
    assert "NEAR_MISS" in t and "PERSON_VEHICLE_COLLISION" not in t, t


def test_vehicle_far_behind_no_alert():
    """bbox xe che người trên ảnh nhưng xe ở xa 4 m phía sau ⇒ không báo."""
    n = 40
    ps = static_person(n)
    frames = [[ps[i], vehicle_det(1, 100 + 30 * i, 200, 200, 120)] for i in range(n)]
    assert not run(make_engine(True), frames)


def test_vehicle_hits_person():
    """Xe chạy 3 m/s tông vào người đứng cùng độ sâu, người đổ sau tiếp xúc."""
    rng = np.random.default_rng(3)
    frames, cx, i_contact = [], 300.0, None
    for i in range(60):
        if i_contact is None:
            cx += 30
            if cx + 100 >= 625:
                i_contact = i
        if i_contact is None:
            k = posture_points(STAND, 640, 600, 300, 0, 0.004, rng)
        else:
            j = i - i_contact
            k = posture_points(STAND, 640 + 4 * min(j, 8), 600, 300, min(90, 12 * j), 0.004, rng)
        frames.append([person_det(1, k, 300), vehicle_det(1, cx, 605, 200, 120)])
    t = types(run(make_engine(True), frames))
    assert "PERSON_VEHICLE_COLLISION" in t, t


def test_person_beside_parked_truck_no_alert():
    n = 100
    ps = static_person(n)
    frames = [[ps[i], vehicle_det(1, 520, 605, 200, 160, "truck")] for i in range(n)]
    assert not run(make_engine(True), frames)


def test_vehicle_vehicle_collision():
    frames, xa, xb = [], 200.0, 1080.0
    stopped = False
    for i in range(50):
        if not stopped:
            xa, xb = xa + 30, xb - 30
            if xb - xa <= 200:
                stopped = True
        frames.append([vehicle_det(1, xa, 400, 200, 120), vehicle_det(2, xb, 400, 200, 120)])
    t = types(run(make_engine(True), frames))
    assert t.count("VEHICLE_COLLISION") == 1, t


def test_parallel_vehicles_no_collision():
    frames = [[vehicle_det(1, 100 + 30 * i, 400, 200, 120), vehicle_det(2, 100 + 30 * i, 445, 200, 120)]
              for i in range(35)]
    t = types(run(make_engine(True), frames))
    assert "VEHICLE_COLLISION" not in t, t


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
