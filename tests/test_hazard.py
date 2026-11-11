"""Cháy / khói: xác nhận theo thời gian, bộ lọc màu dự phòng, và đọc kết quả YOLO (stub, không cần torch)."""
import sys
import types as pytypes

import cv2
import numpy as np

from tests.sim import STAND, person_det, posture_points
from tests.test_scenarios import H_IMG, W, make_engine, types

BOX = np.array([400.0, 300.0, 520.0, 460.0])


def feed(eng, kinds_per_frame, dt=0.3):
    """kinds_per_frame: list các list [(kind, conf, box)] – mỗi phần tử là một lần mô hình chạy."""
    ev = []
    for i, hz in enumerate(kinds_per_frame):
        ev += [e for e in eng.process(i * dt, [], None, hz) if "type" in e]
    return ev


def test_fire_confirmed_after_min_hits():
    ev = feed(make_engine(), [[("fire", 0.6, BOX + j)] for j in range(5)])
    assert types(ev) == ["FIRE_SMOKE"] and ev[0]["severity"] == "critical", types(ev)
    assert ev[0]["details"]["kind"] == "fire" and ev[0]["details"]["source"] == "color"  # model_active chưa bật


def test_single_or_scattered_detections_not_enough():
    assert not feed(make_engine(), [[("fire", 0.9, BOX)]] + [[]] * 6)
    scattered = [[("fire", 0.9, np.array([50 + 250 * j, 100.0, 120 + 250 * j, 190.0]))] for j in range(5)]
    assert not feed(make_engine(), scattered)             # mỗi lần một chỗ khác nhau: không đủ cùng vị trí


def test_smoke_is_warning_and_model_source():
    eng = make_engine()
    eng.hazard.model_active = True
    ev = feed(eng, [[("smoke", 0.5, BOX)] for _ in range(4)])
    assert types(ev) == ["FIRE_SMOKE"] and ev[0]["severity"] == "warning"
    assert ev[0]["details"]["source"] == "model"


def test_cooldown_prevents_repeated_alerts():
    ev = feed(make_engine(), [[("fire", 0.6, BOX)] for _ in range(40)])      # 12 s liên tục
    assert types(ev) == ["FIRE_SMOKE"], types(ev)                              # cooldown 30 s


def test_window_expiry():
    """Phát hiện cách nhau quá cửa sổ 3 s không được cộng dồn."""
    seq = [[("fire", 0.6, BOX)], [], [], [], [], [], [], [], [], [], [("fire", 0.6, BOX)], [],
           [], [], [], [], [], [], [], [], [], [("fire", 0.6, BOX)]]
    assert not feed(make_engine(), seq)


# ------------------------------------------------------------ bộ lọc màu dự phòng
def scene(i, flame=None, vest=None, person_box=None):
    img = np.full((H_IMG, W, 3), 40, np.uint8)
    rng = np.random.default_rng(i)
    img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    if flame == "flicker":
        r = 45 + rng.integers(-18, 19, 2)
        c = (500 + int(rng.integers(-15, 16)), 400 + int(rng.integers(-15, 16)))
        cv2.ellipse(img, c, (int(r[0]), int(r[1] * 1.6)), float(rng.integers(0, 180)), 0, 360, (0, 140, 255), -1)
    if vest:
        cv2.ellipse(img, (500, 400), (45, 70), 0, 0, 360, (0, 140, 255), -1)
    return img


def run_color(frames_fn, dets_fn=lambda i: [], n=40):
    eng, ev = make_engine(), []
    for i in range(n):
        ev += [e for e in eng.process(i * 0.1, dets_fn(i), frames_fn(i)) if "type" in e]
    return ev


def test_color_fallback_detects_flickering_flame():
    ev = run_color(lambda i: scene(i, flame="flicker"))
    assert types(ev) == ["FIRE_SMOKE"] and ev[0]["details"]["source"] == "color", types(ev)
    x1, y1, x2, y2 = ev[0]["boxes"][0]
    assert 380 < x1 < 520 < x2 < 640 and 250 < y1 < 400 < y2 < 560      # khoanh đúng chỗ


def test_color_fallback_ignores_static_orange_object():
    assert not run_color(lambda i: scene(i, vest=True))


def test_color_fallback_ignores_orange_on_person():
    """Áo phản quang cam trên người đang đi lại (nhấp nháy theo dáng đi) không phải lửa."""
    rng = np.random.default_rng(1)

    def dets(i):
        k = posture_points(STAND, 500, 520, 300, 0, 0.004, rng)
        return [person_det(1, k, 300)]
    ev = run_color(lambda i: scene(i, flame="flicker"), dets)
    assert not ev, types(ev)


def test_color_fallback_off_when_model_active_or_disabled():
    def go(configure):
        eng, ev = make_engine(), []
        configure(eng)
        for i in range(40):
            ev += [e for e in eng.process(i * 0.1, [], scene(i, flame="flicker")) if "type" in e]
        return ev
    assert not go(lambda e: setattr(e.hazard, "model_active", True))
    assert not go(lambda e: setattr(e.settings.hazard, "color_fallback", False))
    assert not go(lambda e: setattr(e.settings.hazard, "enabled", False))


# ------------------------------------------------------------ đọc kết quả YOLO bằng stub
class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class _Boxes:
    def __init__(self):
        self.xyxy, self.conf, self.cls = _T([[10, 10, 50, 60], [100, 100, 200, 220], [5, 5, 9, 9]]), \
            _T([0.9, 0.3, 0.8]), _T([1, 0, 2])
    def __len__(self): return 3


class _FakeYOLO:
    def __init__(self, path):
        self.path = str(path)
        self.names = {0: "smoke", 1: "fire", 2: "cone"} if "fire" in self.path else \
                     ({0: "person"} if "pose" in self.path else {2: "car"})
    def predict(self, frame, **kw):
        return [pytypes.SimpleNamespace(boxes=_Boxes())]
    def track(self, *a, **k):
        return [pytypes.SimpleNamespace(boxes=None, keypoints=None)]


def _detector(monkeypatch, fire_model, tmp_path):
    from app.config import Settings
    from app import pipeline
    monkeypatch.setitem(sys.modules, "ultralytics", pytypes.SimpleNamespace(YOLO=_FakeYOLO))
    s = Settings()
    s.pipeline.pose_model, s.pipeline.det_model = "pose.pt", "det.pt"
    s.hazard.fire_model = fire_model
    return pipeline.Detector(s)


def test_detector_parses_fire_model_output(monkeypatch, tmp_path):
    f = tmp_path / "fire_smoke_best.pt"
    f.write_bytes(b"x")
    d = _detector(monkeypatch, str(f), tmp_path)
    assert d.fire is not None and d.fire_map == {0: "smoke", 1: "fire"}      # lớp 'cone' bị bỏ
    out = d.detect_fire(np.zeros((100, 100, 3), np.uint8))
    assert [(k, round(c, 1)) for k, c, _ in out] == [("fire", 0.9), ("smoke", 0.3)]


def test_detector_missing_or_wrong_model_falls_back(monkeypatch, tmp_path):
    assert _detector(monkeypatch, str(tmp_path / "khong_co.pt"), tmp_path).fire is None
    other = tmp_path / "other.pt"
    other.write_bytes(b"x")                                                     # tên không có fire ⇒ lớp 'person'
    assert _detector(monkeypatch, str(other), tmp_path).fire is None
    assert _detector(monkeypatch, "", tmp_path).fire is None
