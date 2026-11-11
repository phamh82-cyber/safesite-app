"""Kiểm thử LiveReader bằng nguồn camera giả (không cần camera/mạng thật)."""
import threading
import time

import numpy as np
import pytest

from app import pipeline
from app.pipeline import LiveReader


class FakeCap:
    """Giả lập cv2.VideoCapture: phát khung đánh số 1, 2, 3… ở tốc độ fps.
    Sau `fail_after` khung thì read() trả về False (mất kết nối)."""

    def __init__(self, fps=100.0, fail_after=None, stall=False):
        self.fps, self.fail_after, self.stall = fps, fail_after, stall
        self.n = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 0.0

    def read(self):
        if self.stall:                    # camera treo: không có khung nào
            time.sleep(0.05)
            return False, None
        if self.released or (self.fail_after is not None and self.n >= self.fail_after):
            return False, None
        time.sleep(1.0 / self.fps)
        self.n += 1
        return True, np.array([self.n])

    def release(self):
        self.released = True


def use_caps(monkeypatch, *caps):
    """open_source trả lần lượt các FakeCap; ghi lại số lần được gọi."""
    queue, opened = list(caps), []

    def fake_open(source):
        cap = queue.pop(0) if len(queue) > 1 else queue[0]
        opened.append(cap)
        return cap

    monkeypatch.setattr(pipeline, "open_source", fake_open)
    return opened


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_returns_latest_frame_not_queued(monkeypatch):
    # Camera 100 FPS, phân tích chậm 200 ms/khung: phải nhận khung mới nhất, bỏ khung cũ
    cap = FakeCap(fps=100)
    use_caps(monkeypatch, cap)
    r = LiveReader("http://cam/video")
    try:
        seq, frame, _ = r.read(0, timeout=1.0)
        assert frame is not None
        for _ in range(3):
            time.sleep(0.2)
            prev = seq
            seq, frame, t = r.read(seq, timeout=1.0)
            assert seq - prev >= 10                      # đã bỏ qua các khung cũ
            assert int(frame[0]) == seq                  # khung trả về đúng là khung mới nhất
            assert cap.n - seq <= 2
            assert time.monotonic() - t < 0.05           # khung còn "tươi"
    finally:
        r.release()


def test_each_frame_returned_once(monkeypatch):
    use_caps(monkeypatch, FakeCap(fps=50))
    r = LiveReader("http://cam/video")
    try:
        seqs, seq = [], 0
        for _ in range(10):
            seq, frame, _ = r.read(seq, timeout=1.0)
            assert frame is not None
            seqs.append(seq)
        assert seqs == sorted(set(seqs))                 # không trả lặp một khung
    finally:
        r.release()


def test_timeout_when_camera_stalls(monkeypatch):
    use_caps(monkeypatch, FakeCap(stall=True))
    r = LiveReader("http://cam/video")
    try:
        t0 = time.monotonic()
        seq, frame, _ = r.read(0, timeout=0.2)
        assert frame is None and seq == 0
        assert 0.15 < time.monotonic() - t0 < 1.0        # không treo lâu hơn timeout
        assert not r.dead                                # camera mạng: vẫn đang thử lại
    finally:
        r.release()


def test_network_camera_reconnects(monkeypatch):
    monkeypatch.setattr(LiveReader, "retry_delay", 0.05)
    first, second = FakeCap(fps=100, fail_after=3), FakeCap(fps=100)
    opened = use_caps(monkeypatch, first, second)
    r = LiveReader("rtsp://cam/stream")
    try:
        assert wait_until(lambda: second.n >= 5)         # đã mở lại và nhận hình từ kết nối mới
        assert first.released
        assert len(opened) >= 2
        assert not r.dead
        seq, frame, _ = r.read(0, timeout=1.0)
        assert frame is not None and seq > 3
    finally:
        r.release()


class ClosedCap(FakeCap):
    def isOpened(self):
        return False

    def read(self):
        return False, None


def test_network_camera_retries_when_first_open_fails(monkeypatch):
    # Bấm Chạy lúc điện thoại chưa phát hình: không báo lỗi ngay, tự lên hình khi có
    monkeypatch.setattr(LiveReader, "retry_delay", 0.05)
    later = FakeCap(fps=100)
    use_caps(monkeypatch, ClosedCap(), ClosedCap(), later)
    r = LiveReader("http://cam/video")
    try:
        assert r.isOpened()
        assert wait_until(lambda: later.n >= 3)
        seq, frame, _ = r.read(0, timeout=1.0)
        assert frame is not None and not r.dead
    finally:
        r.release()


def test_local_webcam_not_opened(monkeypatch):
    use_caps(monkeypatch, ClosedCap())
    r = LiveReader("0")
    assert not r.isOpened()                              # webcam USB không có -> báo lỗi ngay


def test_local_webcam_lost_marks_dead(monkeypatch):
    # Webcam USB (nguồn "0") mất hình: không kết nối lại, báo dead để worker báo lỗi
    cap = FakeCap(fps=100, fail_after=2)
    opened = use_caps(monkeypatch, cap)
    r = LiveReader("0")
    try:
        assert wait_until(lambda: r.dead)
        assert len(opened) == 1 and cap.released
        seq, frame, _ = r.read(2, timeout=1.0)
        assert frame is None                             # trả về ngay, không chờ hết timeout
    finally:
        r.release()


def test_release_stops_reader_thread(monkeypatch):
    cap = FakeCap(fps=100)
    use_caps(monkeypatch, cap)
    before = threading.active_count()
    r = LiveReader("http://cam/video")
    assert wait_until(lambda: cap.n > 0)
    r.release()
    assert wait_until(lambda: cap.released and r.dead)
    assert wait_until(lambda: threading.active_count() <= before)


@pytest.mark.parametrize("src,expected", [
    ("http://192.168.1.5:8080/video", True), ("rtsp://u:p@10.0.0.2:554/s", True),
    ("0", False), (r"E:\videos\a.mp4", False), (0, False)])
def test_is_network(src, expected):
    assert pipeline.is_network(src) is expected
