"""Smoke test không cần GPU/model: detector và VideoMAE giả lập, chạy pipeline thật trên video tổng hợp.
Chạy:  python -m tests.test_pipeline
"""
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from app.analyzers.yolo_detector import Det, calc_torso_angle
from app.buffers.clip_buffer import union_roi
from app.config import CameraConfig, SettingsStore
from app.engine import Engine
from app.managers.event_manager import EventStore
from app.trackers.track_manager import Candidate


def make_video(path, n=150, fps=25):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    for i in range(n):
        f = np.full((360, 640, 3), 60, np.uint8)
        cv2.putText(f, str(i), (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        vw.write(f)
    vw.release()


class FakeDetector:
    device = "cpu"

    def __init__(self):
        self.n = 0

    def detect(self, frame, s):
        self.n += 1
        if self.n < 30:   # đứng
            k = np.array([[0, 0, 0]] * 5 + [[300, 100, 1], [320, 100, 1]] + [[0, 0, 0]] * 4
                         + [[300, 200, 1], [320, 200, 1]] + [[0, 0, 0]] * 4, float)
            box = (280, 80, 340, 280)
        else:             # nằm: vai lệch ngang so với hông
            k = np.array([[0, 0, 0]] * 5 + [[200, 250, 1], [200, 270, 1]] + [[0, 0, 0]] * 4
                         + [[300, 250, 1], [300, 270, 1]] + [[0, 0, 0]] * 4, float)
            box = (180, 230, 360, 290)
        return [Det(box, 0.9, "person", k)], [], []


class FakeVM:
    device, fp16 = "cpu", False

    def __init__(self):
        self.calls = 0

    def predict(self, views):
        self.calls += 1
        assert len(views) == 2 and len(views[0]) == 16 and len(views[1]) == 16
        return {"hazard_prob": 0.8, "top_action": "person falling down", "action_conf": 0.9,
                "view": "roi", "per_view": [0.5, 0.8]}


def test_geometry():
    up = np.zeros((17, 3)); up[5] = up[6] = (100, 100, 1); up[11] = up[12] = (100, 200, 1)
    lay = np.zeros((17, 3)); lay[5] = lay[6] = (100, 100, 1); lay[11] = lay[12] = (200, 110, 1)
    assert calc_torso_angle(up) < 5 and calc_torso_angle(lay) > 80
    assert calc_torso_angle(None, (0, 0, 100, 40)) == 70.0
    roi = union_roi([(100, 100, 200, 200)], (360, 640, 3), 0.3)
    assert roi[2] - roi[0] == roi[3] - roi[1]
    print("geometry OK")


def test_pipeline():
    tmp = Path(tempfile.mkdtemp())
    vid = tmp / "v.mp4"
    make_video(vid)
    store = SettingsStore(tmp / "settings.json")
    store.update_settings({"clip_pre_s": 1.0, "clip_post_s": 1.0, "cooldown_s": 10})
    store.set_cameras([CameraConfig("c1", "Cam 1", str(vid), "Khu A")])
    msgs = []
    eng = Engine(store, msgs.append)
    eng.events.db = EventStore(tmp / "test_events.db")
    eng.detector, eng.videomae = FakeDetector(), FakeVM()
    eng.reload_cameras()
    t0 = time.time()
    while time.time() - t0 < 9:
        time.sleep(0.5)
        if any(m["type"] == "event_update" for m in msgs):
            break
    pipe = eng.pipelines["c1"]
    st = eng.status()["cameras"][0]
    eng.stop()
    evs = eng.events.db.list()
    print("status:", st)
    print("events:", [(e["id"], e["label"], round(e["confidence"], 2), e["clip"], e["snapshot"]) for e in evs])
    assert pipe.latest_jpeg
    assert len(evs) == 1, "cooldown phải chặn sự cố lặp"
    assert evs[0]["label"] == "person falling down" and evs[0]["clip"], "thiếu clip"
    from app.config import CLIPS_DIR, SNAPSHOTS_DIR
    clip = CLIPS_DIR / evs[0]["clip"]
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print("clip frames:", n)
    assert n > 10 and (SNAPSHOTS_DIR / evs[0]["snapshot"]).exists()
    clip.unlink(); (SNAPSHOTS_DIR / evs[0]["snapshot"]).unlink()
    assert [m["type"] for m in msgs if m["type"].startswith("event")][:2] == ["event", "event_update"]
    print("pipeline OK")


def test_scan_hazard():
    from app.config import Settings
    from app.managers.event_manager import EventManager
    store = SettingsStore(Path(tempfile.mkdtemp()) / "settings.json")
    ev_mgr = EventManager(store)
    s = store.settings
    # Scan candidate with VideoMAE predicting some action
    cand_scan = Candidate("scan", "scan", [], [], 0.0)
    vm_res = {"hazard_prob": 0.85, "top_action": "collision between vehicles", "action_conf": 0.9}
    ok, label, conf = ev_mgr.fuse(cand_scan, vm_res, s)
    assert ok and label == "hazard", f"Expected 'hazard' for scan, got {label}"
    
    # Fall candidate
    cand_fall = Candidate("fall", "fall:1", [1], [(0,0,10,10)], 0.8)
    ok2, label2, conf2 = ev_mgr.fuse(cand_fall, vm_res, s)
    assert ok2 and label2 == "collision between vehicles"
    print("test_scan_hazard OK")


def test_pause_resume():
    tmp = Path(tempfile.mkdtemp())
    vid = tmp / "v_pause.mp4"
    make_video(vid, n=50)
    store = SettingsStore(tmp / "settings.json")
    store.set_cameras([CameraConfig("cp1", "Cam Pause", str(vid), "Khu B")])
    eng = Engine(store)
    eng.events.db = EventStore(tmp / "pause_events.db")
    eng.reload_cameras()
    time.sleep(0.5)
    pipe = eng.pipelines["cp1"]
    assert not pipe.is_paused()

    pipe.set_paused(True)
    assert pipe.is_paused()
    st = pipe.status()
    assert st["paused"] is True and st["fps"] == 0.0

    pipe.toggle_paused()
    assert not pipe.is_paused()
    st2 = pipe.status()
    assert st2["paused"] is False

    eng.stop()
    print("test_pause_resume OK")


if __name__ == "__main__":
    test_geometry()
    test_scan_hazard()
    test_pause_resume()
    test_pipeline()
