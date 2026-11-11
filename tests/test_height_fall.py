"""Kiểm thử ngã từ trên cao (FALL_FROM_HEIGHT) bằng chuyển động giả lập."""
from __future__ import annotations

import numpy as np

from tests.sim import STAND, person_det, posture_points
from tests.test_scenarios import FPS, make_engine, person_seq, run, types

HP = 200            # chiều cao người (px)
FX, FY0 = 640, 250  # chân đứng trên "sàn cao"


def seq(segments, tid=1, seed=0):
    """segments: (thời lượng, fn(s) -> (tư thế, góc đổ, fy, chiều cao px)); s ∈ [0,1]."""
    rng = np.random.default_rng(seed)
    out = []
    for dur, fn in segments:
        n = max(1, int(round(dur * FPS)))
        for j in range(n):
            post, topple, fy, hp = fn(j / max(n - 1, 1))
            k = posture_points(post, FX, fy, hp, topple, jitter=0.004, rng=rng)
            out.append([person_det(tid, k, hp)])
    return out


def stand(dur, fy=FY0):
    return (dur, lambda s: (STAND, 0, fy, HP))


def free_fall(drop_h, fy0=FY0):
    """Rơi tự do `drop_h` chiều cao người, vẫn đứng thẳng."""
    T = (2 * drop_h * 1.7 / 9.81) ** 0.5
    return (T, lambda s: (STAND, 0, fy0 + drop_h * HP * s * s, HP))


def land_and_lie(fy, dur=3.0):
    return [(0.3, lambda s: (STAND, 90 * s, fy, HP)), (dur, lambda s: (STAND, 90, fy, HP))]


def test_fall_from_height_then_lying():
    fy_end = FY0 + 2.0 * HP
    ev = run(make_engine(), seq([stand(2.0), free_fall(2.0), *land_and_lie(fy_end)]))
    hf = [e for e in ev if e["type"] == "FALL_FROM_HEIGHT"]
    assert len(hf) == 1, types(ev)
    assert 1.5 <= hf[0]["details"]["est_drop_m"] <= 5, hf[0]["details"]
    assert "FALL" not in types(ev), "báo trùng té ngã thường cho cùng một vụ"


def test_ground_fall_is_not_height_fall():
    frames = person_seq([(2.0, lambda s: (STAND, 0)), (0.8, lambda s: (STAND, 90 * s ** 1.5)),
                         (4.0, lambda s: (STAND, 90))])
    t = types(run(make_engine(), frames))
    assert "FALL" in t and "FALL_FROM_HEIGHT" not in t, t


def test_low_jump_down_then_ground_fall():
    """Nhảy xuống bậc 0,6 chiều cao người rồi vấp ngã: chỉ là té ngã thường."""
    ev = run(make_engine(), seq([stand(2.0), free_fall(0.6), stand(1.5, FY0 + 0.6 * HP),
                                 *land_and_lie(FY0 + 0.6 * HP)]))
    assert types(ev) == ["FALL"], types(ev)


def test_low_jump_down_standing_no_alert():
    ev = run(make_engine(), seq([stand(2.0), free_fall(0.6), stand(4.0, FY0 + 0.6 * HP)]))
    assert not ev, types(ev)


def test_deep_drop_but_lands_standing_no_alert():
    """Rơi 1,0 chiều cao người nhưng hạ cánh đứng và đi tiếp: vào pha rơi rồi hủy, không báo."""
    ev = run(make_engine(), seq([stand(2.0), free_fall(1.0), stand(5.0, FY0 + 1.0 * HP)]))
    assert "FALL_FROM_HEIGHT" not in types(ev), types(ev)


def test_running_toward_camera_then_trip_is_ground_fall():
    """Bbox to dần 2,2 lần khi tiến sát camera: hông tụt nhanh trên ảnh nhưng không phải rơi.
    Cuối cùng người vấp ngã nằm ⇒ chỉ được là té ngã thường."""
    def run_in(s):
        return STAND, 0, 300 + 350 * s, 150 + 170 * s
    ev = run(make_engine(), seq([(1.5, lambda s: (STAND, 0, 300, 150)), (1.0, run_in),
                                 (0.3, lambda s: (STAND, 90 * s, 650, 320)),
                                 (3.0, lambda s: (STAND, 90, 650, 320))]))
    t = types(ev)
    assert "FALL_FROM_HEIGHT" not in t, t
    assert "FALL" in t, t


def test_slow_descent_stairs_then_lie_down_not_height_fall():
    ev = run(make_engine(), seq([stand(2.0), (3.0, lambda s: (STAND, 0, FY0 + 1.5 * HP * s, HP)),
                                 stand(1.0, FY0 + 1.5 * HP), *land_and_lie(FY0 + 1.5 * HP)]))
    assert "FALL_FROM_HEIGHT" not in types(ev), types(ev)


def test_falls_out_of_frame_bottom():
    """Rơi ra khỏi mép dưới khung hình: mất dấu khi đang rơi nhanh."""
    frames = seq([stand(2.0), free_fall(2.2, fy0=FY0 - 50)])
    frames = [f for f in frames if f[0].box[3] < 715] + [[] for _ in range(25)]
    ev = run(make_engine(), frames)
    hf = [e for e in ev if e["type"] == "FALL_FROM_HEIGHT"]
    assert len(hf) == 1 and hf[0]["confidence"] < 0.8, types(ev)


def test_id_switch_after_landing_one_alert():
    fy_end = FY0 + 2.0 * HP
    a = seq([stand(2.0), free_fall(2.0)], tid=1)
    b = seq(land_and_lie(fy_end), tid=5)           # tracker cấp ID mới khi người đã nằm
    ev = run(make_engine(), a + b)
    assert types(ev).count("FALL_FROM_HEIGHT") == 1 and "FALL" not in types(ev), types(ev)


def test_fall_from_height_at_low_fps(monkeypatch):
    """5 FPS (CPU yếu): cửa sổ tính vận tốc tự nới nên vẫn bắt được."""
    import tests.test_scenarios as ts
    import tests.test_height_fall as me
    monkeypatch.setattr(ts, "FPS", 5.0); monkeypatch.setattr(ts, "DT", 0.2)
    monkeypatch.setattr(me, "FPS", 5.0)
    fy_end = FY0 + 2.0 * HP
    ev = ts.run(ts.make_engine(), me.seq([me.stand(2.0), me.free_fall(2.0), *me.land_and_lie(fy_end)]))
    assert "FALL_FROM_HEIGHT" in types(ev), types(ev)
