"""Bù rung camera: đo đúng dịch chuyển, bỏ qua vật chuyển động, và giúp vận tốc xe đỗ về 0."""
import cv2
import numpy as np

from app.ego import EgoMotion
from tests.sim import vehicle_det
from tests.test_objects import make_bg, render
from tests.test_scenarios import H_IMG, W, make_engine


def shifted(bg, dx, dy, rng):
    return render(bg, rng, shift=(dx, dy))


def test_measures_known_shift():
    bg, rng = make_bg(), np.random.default_rng(0)
    ego = EgoMotion(W, H_IMG, leak=1.0)
    ego.update(shifted(bg, 0, 0, rng))
    for k in range(1, 6):
        ego.update(shifted(bg, 3 * k, -2 * k, rng))       # +3, −2 px mỗi khung
    assert ego.ok and abs(ego.off[0] - 15) < 1.5 and abs(ego.off[1] + 10) < 1.5, ego.off


def test_ignores_moving_object_when_masked():
    """Vật to chuyển động trong khung không được kéo ước lượng lệch khỏi 0 (camera đứng yên)."""
    bg, rng = make_bg(), np.random.default_rng(1)
    ego = EgoMotion(W, H_IMG, leak=1.0)
    for i in range(8):
        x = 200 + 60 * i
        img = render(bg, rng, rects=[(x, 200, x + 300, 520, 20)])
        ego.update(img, boxes=[(x, 200, x + 300, 520)])
    assert np.linalg.norm(ego.off) < 2.0, ego.off


def test_static_scene_reports_no_shake():
    bg, rng = make_bg(), np.random.default_rng(2)
    ego = EgoMotion(W, H_IMG)
    for _ in range(10):
        ego.update(render(bg, rng))
    assert ego.level < 0.001


def _shaky_parked_vehicle(stabilize: bool, amp: float = 12.0, cm_per_px: float = 1.0):
    """Xe đỗ + người đứng cạnh, camera rung ngẫu nhiên (σ = amp px). Trả (loại sự kiện, tốc độ xe ước lượng)."""
    from app.config import Settings
    from app.engine import Engine
    from tests.sim import STAND, person_det, posture_points
    from tests.test_scenarios import types
    bg, rng, prng = make_bg(), np.random.default_rng(4), np.random.default_rng(9)
    eng = Engine(Settings(), W, H_IMG)
    eng.settings.pipeline.stabilize = stabilize
    gw, gh = W * cm_per_px / 100, H_IMG * cm_per_px / 100
    eng.set_calibration([[0, 0], [W, 0], [W, H_IMG], [0, H_IMG]], [[0, 0], [gw, 0], [gw, gh], [0, gh]])
    ev, shake = [], np.zeros(2)
    for i in range(80):
        shake = 0.6 * shake + rng.normal(0, amp, 2)
        dx, dy = shake
        k = posture_points(STAND, 760 + dx, 560 + dy, 200, 0, 0.004, prng)
        dets = [vehicle_det(1, 600 + dx, 520 + dy, 200, 120), person_det(1, k, 200)]
        ev += [e for e in eng.process(i * 0.1, dets, render(bg, rng, shift=(dx, dy))) if "type" in e]
    return types(ev), eng.tm.tracks["V1"].speed()


def test_shaky_camera_parked_vehicle_no_false_near_miss():
    """Không bù rung: xe đỗ bị coi là đang chạy ⇒ NEAR_MISS giả với người đứng cạnh. Có bù rung: không."""
    on_ev, on_v = _shaky_parked_vehicle(True)
    off_ev, off_v = _shaky_parked_vehicle(False)
    assert "NEAR_MISS" not in on_ev and on_v < 0.25, (on_ev, on_v)
    assert "NEAR_MISS" in off_ev and off_v > 0.4, (off_ev, off_v)       # chứng minh bù rung có tác dụng


def test_shake_speed_reduced_when_far_camera():
    """Camera nhìn xa (5 cm/px): tốc độ ảo > 1 m/s nếu không bù, ≤ 0,3 m/s nếu có bù."""
    assert _shaky_parked_vehicle(True, 12, 5.0)[1] < 0.3
    assert _shaky_parked_vehicle(False, 12, 5.0)[1] > 1.0
