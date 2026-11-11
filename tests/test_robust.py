"""Độ bền: hiệu chỉnh khi đang phân tích (race), worker tự phục hồi sau lỗi, clip khi video lặp / dừng."""
import threading
import time

import cv2
import numpy as np
import pytest

from app import pipeline
from app.config import Settings
from app.events import ClipRecorder
from tests.sim import STAND, person_det, posture_points, vehicle_det
from tests.test_scenarios import H_IMG, W, make_engine


def test_calibration_while_processing_does_not_crash_worker():
    """Trước khi sửa: set_calibration từ luồng API làm luồng phân tích chết (IndexError) sau vài lần."""
    eng, errs, stop = make_engine(), [], threading.Event()

    def worker():
        rng, i = np.random.default_rng(0), 0
        try:
            while not stop.is_set():
                dets = [person_det(p + 1, posture_points(STAND, 100 + p * 180 + i % 50, 650, 300, 0, 0.004, rng), 300)
                        for p in range(6)]
                dets.append(vehicle_det(1, 600 + i % 80, 700, 200, 120))
                eng.process(i * 0.1, dets)
                i += 1
        except Exception as e:      # noqa: BLE001
            errs.append(repr(e))

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.5)
    t0, n = time.time(), 0
    while time.time() - t0 < 3 and not errs:
        eng.set_calibration([[0, 0], [W, 0], [W, H_IMG], [0, H_IMG]], [[0, 0], [12.8, 0], [12.8, 7.2], [0, 7.2]])
        n += 1
    stop.set()
    th.join()
    assert n > 20 and not errs, (n, errs)


# ----------------------------------------------------------------- worker tự phục hồi
class FakeHub:
    def emit(self, *a, **k): pass
    def recent_for(self, cid): return []


def make_video(path, n=40, w=320, h=240):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n):
        wr.write(np.full((h, w, 3), 30 + i, np.uint8))
    wr.release()


def make_worker(tmp_path, monkeypatch, fail):
    vid = tmp_path / "v.mp4"
    make_video(vid)
    calls = {"n": 0, "made": 0}

    class FakeDetector:
        fire = None

        def __init__(self, s):
            calls["made"] += 1

        def __call__(self, frame):
            calls["n"] += 1
            if fail(calls["n"]):
                raise RuntimeError("lỗi giả lập")
            return []

    monkeypatch.setattr(pipeline, "Detector", FakeDetector)
    s = Settings()
    s.pipeline.realtime_files = False
    s.pipeline.process_fps = 10
    w = pipeline.CameraWorker({"id": "c1", "name": "t", "source": str(vid), "source_type": "file"}, FakeHub(), s)
    w.restart_backoff = 0.05
    return w, calls


def test_transient_frame_errors_do_not_kill_worker(tmp_path, monkeypatch):
    w, calls = make_worker(tmp_path, monkeypatch, fail=lambda n: n in (3, 4))     # 2 khung lỗi, không liên tiếp ≥3
    w.start()
    w.join(10)
    assert not w.is_alive() and w.status == "finished" and w.restarts == 0, (w.status, w.error)
    assert calls["n"] == 40                                                        # vẫn xử lý hết video


def test_persistent_errors_restart_pipeline_then_stop(tmp_path, monkeypatch):
    w, calls = make_worker(tmp_path, monkeypatch, fail=lambda n: True)
    w.start()
    deadline = time.time() + 10
    while w.restarts < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert w.restarts >= 2 and "tự chạy lại" in w.error, (w.restarts, w.error)
    assert calls["made"] >= w.restarts                                             # mỗi lần chạy lại dựng detector mới
    w.stop()
    w.join(5)
    assert not w.is_alive()


def test_recovers_after_errors_and_finishes(tmp_path, monkeypatch):
    """Lỗi liên tiếp ở lần chạy đầu ⇒ chạy lại ⇒ lần sau ổn định, chạy hết video."""
    w, calls = make_worker(tmp_path, monkeypatch, fail=lambda n: n <= 3)
    w.start()
    w.join(10)
    assert not w.is_alive() and w.status == "finished" and w.restarts == 1, (w.status, w.restarts, w.error)


# ----------------------------------------------------------------- clip
def test_clip_buffer_resets_when_video_loops():
    cr = ClipRecorder(3.0, 3.0)
    f = np.zeros((48, 64, 3), np.uint8)
    for _ in range(3):                       # video 10 s lặp 3 lần, đồng hồ quay về 0 mỗi vòng
        for i in range(100):
            cr.push(i * 0.1, f)
    assert 25 <= len(cr.buf) <= 35, len(cr.buf)       # ≈ 3 s × 10 khung (trước sửa: 231)


def test_clip_flush_writes_pending_when_stream_ends(tmp_path):
    cr = ClipRecorder(2.0, 5.0)
    f = np.zeros((48, 64, 3), np.uint8)
    done = []
    for i in range(20):
        cr.push(i * 0.1, f)
    cr.request(1.9, tmp_path / "c.webm", done.append)    # cần thêm 5 s nhưng luồng dừng ngay
    cr.flush()
    for _ in range(100):
        if done:
            break
        time.sleep(0.05)
    assert done, "clip dở dang bị mất khi camera dừng"
