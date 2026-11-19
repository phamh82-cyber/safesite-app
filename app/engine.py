"""Luồng xử lý trung tâm: mỗi camera 1 pipeline (thread) + 1 thread xác thực VideoMAE bất đồng bộ.

Cascade: YOLO chạy mọi frame -> ByteTrack -> Candidate (góc ngã / gần xe / lửa)
         -> chỉ khi có nghi ngờ mới gom 16 frame (full + ROI) cho VideoMAE -> Event Manager (Fusion).
"""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.buffers.clip_buffer import ClipBuffer, EvidenceBuffer, encode_jpeg, union_roi
from app.config import (ACTION_VOCAB, VIDEOMAE_WEIGHTS, YOLO_OBJ_WEIGHTS, YOLO_POSE_WEIGHTS,
                        CameraConfig, SettingsStore)
from app.managers.event_manager import EventManager
from app.trackers.bytetrack_wrapper import ByteTrackWrapper
from app.trackers.track_manager import Candidate, TrackManager

GREEN, YELLOW, RED = (0, 200, 0), (0, 215, 255), (0, 0, 255)


# ----------------------------------------------------------------------------- đọc frame
class FrameGrabber(threading.Thread):
    """Đọc camera/file ở thread riêng, luôn giữ frame mới nhất (RTSP không bị dồn trễ)."""

    def __init__(self, source: str):
        super().__init__(daemon=True)
        self.source = str(source).strip()
        self.stop_ev = threading.Event()
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.connected = False
        self.error = ""
        self.paused = False
        self.ended = False

    def _open(self):
        return cv2.VideoCapture(int(self.source)) if self.source.isdigit() else cv2.VideoCapture(self.source)

    def set_paused(self, paused: bool):
        with self.cond:
            self.paused = bool(paused)
            self.cond.notify_all()

    def is_paused(self) -> bool:
        with self.cond:
            return self.paused

    def run(self):
        is_file = Path(self.source).is_file()
        cap, fps, next_t, fails = None, 25.0, 0.0, 0
        while not self.stop_ev.is_set():
            if self.is_paused():
                next_t = time.time()
                self.stop_ev.wait(0.05)
                continue
            if self.ended:
                self.stop_ev.wait(0.05)
                continue
            if cap is None:
                cap = self._open()
                if not cap.isOpened():
                    cap.release()
                    cap, self.connected, self.error = None, False, "không mở được nguồn"
                    self.stop_ev.wait(2.0)
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                fps = 25.0 if fps < 1 or fps > 120 else fps
                self.connected, self.error, next_t, fails = True, "", time.time(), 0
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if is_file:
                    cap.release()
                    cap, self.connected, self.error, self.ended = None, False, "video đã phát hết", True
                    self.stop_ev.wait(1.0)
                    continue
                cap.release()
                cap, self.connected, self.error = None, False, "mất kết nối, đang thử lại"
                self.stop_ev.wait(2.0)
                continue
            fails = 0
            if is_file:                                      # phát đúng tốc độ thật của file
                next_t += 1.0 / fps
                delay = next_t - time.time()
                if delay > 0:
                    self.stop_ev.wait(delay)
                else:
                    next_t = time.time()
            with self.cond:
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()
        if cap is not None:
            cap.release()

    def latest(self, last_seq: int, timeout: float = 1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or self.stop_ev.is_set(), timeout)
            return self.frame, self.seq

    def stop(self):
        self.stop_ev.set()
        with self.cond:
            self.cond.notify_all()


# ----------------------------------------------------------------------------- pipeline 1 camera
class CameraPipeline(threading.Thread):
    def __init__(self, cam: CameraConfig, engine: "Engine"):
        super().__init__(daemon=True, name=f"cam-{cam.id}")
        self.cam, self.engine = cam, engine
        self.grabber = FrameGrabber(cam.source)
        self.tracker_p, self.tracker_v = ByteTrackWrapper(), ByteTrackWrapper()
        self.tm = TrackManager()
        self.buffer = ClipBuffer()
        self.evidence = EvidenceBuffer(max_seconds=10.0)
        self.stop_ev = threading.Event()
        self.latest_jpeg: Optional[bytes] = None
        self.fps = 0.0
        self.hazard_until = 0.0
        self.hazard_prob = 0.0
        self.last_prob: Optional[float] = None
        self.danger_ids: set = set()
        self.last_event_label = ""
        self.last_verify = 0.0
        self.jobs: queue.Queue = queue.Queue(maxsize=1)
        self.verifier = threading.Thread(target=self._verify_loop, daemon=True, name=f"vm-{cam.id}")

    # -------- vòng lặp chính
    def run(self):
        self.grabber.start()
        self.verifier.start()
        seq, last_t = 0, time.time()
        while not self.stop_ev.is_set():
            frame, seq_new = self.grabber.latest(seq)
            if frame is None or seq_new == seq:
                if not self.grabber.connected:
                    self.latest_jpeg = self._offline_jpeg()
                continue
            seq = seq_new
            try:
                self._process(frame)
            except Exception as e:     # không để 1 lỗi làm chết camera
                self.grabber.error = f"lỗi xử lý: {e}"
                print(f"[{self.cam.id}] {type(e).__name__}: {e}")
                time.sleep(0.2)
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last_t, 1e-3) if self.fps else 1 / max(now - last_t, 1e-3)
            last_t = now

    def _process(self, frame):
        s = self.engine.store.settings
        ts = time.time()
        self.buffer.push(ts, frame)

        det = self.engine.detector
        if s.enable_yolo and det is not None:
            persons, vehicles, fires = det.detect(frame, s)
            pt = self._track(self.tracker_p, persons)
            vt = self._track(self.tracker_v, vehicles)
            self.tm.update(ts, s, pt, vt, fires)
            cands = self.tm.candidates(s, ts)
        else:
            self.tm.clear()
            cands = []

        vis = self._draw(frame, ts, s)
        self._maybe_verify(cands, ts, s, vis, frame.shape)

        jpeg = encode_jpeg(vis)
        self.latest_jpeg = jpeg
        self.evidence.push(ts, jpeg)
        self.engine.events.feed_frame(self.cam.id, ts, jpeg)

    @staticmethod
    def _track(tracker, dets):
        if not dets:
            tracker.update(np.zeros((0, 4)), np.zeros(0))
            return []
        out = tracker.update(np.array([d.box for d in dets]), np.array([d.conf for d in dets]))
        return [(row[4], dets[int(row[7])]) for row in out if int(row[7]) < len(dets)]

    # -------- Cascade: chỉ chạy VideoMAE khi có nghi ngờ
    def _maybe_verify(self, cands, ts, s, vis, shape):
        vm = self.engine.videomae
        ev_mgr = self.engine.events
        if not (s.enable_videomae and vm is not None):
            for c in cands:                                   # chỉ dựa hình học
                if c.score >= 0.7 and not ev_mgr.in_cooldown(self.cam.id, c.key, ts):
                    self._emit(c, None, ts, vis)
            return
        if s.enable_yolo and self.engine.detector is not None:
            cands = [c for c in cands if not ev_mgr.in_cooldown(self.cam.id, c.key, ts)]
            if not cands:
                return
            cand = max(cands, key=lambda c: c.score)
        else:                                                 # YOLO tắt: quét định kỳ toàn khung
            cand = Candidate("scan", "scan", [], [], 0.0)
            if ev_mgr.in_cooldown(self.cam.id, cand.key, ts):
                return
        if ts - self.last_verify < s.verify_interval_s or self.jobs.full():
            return
        roi = union_roi(cand.boxes, shape, s.roi_margin)
        full, crop = self.buffer.sample(roi)
        if not full:
            return
        self.last_verify = ts
        self.jobs.put_nowait((cand, [full] + ([crop] if crop else []), ts, vis))

    def _verify_loop(self):
        while not self.stop_ev.is_set():
            try:
                cand, views, ts, vis = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                vm = self.engine.videomae
                res = vm.predict(views)
                self.last_prob = res["hazard_prob"]
                self._emit(cand, res, ts, vis)
            except Exception as e:
                print(f"[{self.cam.id}] VideoMAE lỗi: {type(e).__name__}: {e}")

    def _emit(self, cand, vm_res, ts, vis):
        s = self.engine.store.settings
        now = time.time()
        pre = self.evidence.last(s.clip_pre_s + (now - ts), now)
        ev = self.engine.events.handle(self.cam, cand, vm_res, vis, ts, pre)
        if ev:
            self.hazard_until = now + 5.0
            self.hazard_prob = ev["confidence"]
            self.danger_ids = set(cand.track_ids)
            self.last_event_label = ev["label"]
        return ev

    # -------- vẽ overlay
    def _draw(self, frame, now, s):
        vis = frame.copy()
        h, w = vis.shape[:2]
        sc = max(0.5, w / 1280)
        th = max(1, int(2 * sc))
        hazard = now < self.hazard_until

        def box(b, color, label):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, th)
            if label:
                (tw, tt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, 1)
                cv2.rectangle(vis, (x1, max(0, y1 - tt - 8)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc,
                            (0, 0, 0), 1, cv2.LINE_AA)

        for v in self.tm.vehicles():
            box(v.box, YELLOW, f"{v.name} #{v.id}")
        for p in self.tm.persons():
            danger = p.lying_frames >= s.fall_min_frames or (hazard and p.id in self.danger_ids)
            ang = f" {p.angle:.0f}deg" if p.angle is not None else ""
            box(p.box, RED if danger else GREEN, f"ID {p.id}{ang}")
        for f in self.tm.fires:
            box(f.box, RED, f"{f.name} {f.conf:.2f}")

        bar_w, bar_h = int(330 * sc), int(40 * sc)
        color = RED if hazard and int(now * 3) % 2 == 0 else (0, 0, 160) if hazard else (0, 140, 0)
        cv2.rectangle(vis, (0, 0), (bar_w, bar_h), color, -1)
        if hazard:
            txt = f"HAZARD ALERT {self.hazard_prob * 100:.0f}%"
        else:
            txt = "SAFE" + (f"  (p={self.last_prob * 100:.0f}%)" if self.last_prob is not None else "")
        cv2.putText(vis, txt, (int(10 * sc), int(27 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * sc,
                    (255, 255, 255), th, cv2.LINE_AA)
        cv2.putText(vis, f"{self.cam.name}  {self.fps:.0f} FPS", (int(10 * sc), h - int(10 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def _offline_jpeg(self) -> bytes:
        img = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(img, f"{self.cam.name}: OFFLINE", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, self.grabber.error or "dang ket noi...", (30, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1, cv2.LINE_AA)
        time.sleep(0.5)
        return encode_jpeg(img)

    def is_paused(self) -> bool:
        return self.grabber.is_paused()

    def set_paused(self, paused: bool):
        self.grabber.set_paused(paused)

    def toggle_paused(self) -> bool:
        new_st = not self.grabber.is_paused()
        self.grabber.set_paused(new_st)
        return new_st

    def status(self) -> dict:
        now = time.time()
        paused = self.grabber.paused
        return {"id": self.cam.id, "name": self.cam.name, "zone": self.cam.zone,
                "connected": self.grabber.connected, "error": self.grabber.error,
                "paused": paused,
                "fps": round(self.fps, 1) if not paused else 0.0,
                "hazard": now < self.hazard_until,
                "hazard_prob": round(self.hazard_prob, 3), "last_prob": self.last_prob,
                "persons": len(self.tm.persons()) if not paused else 0,
                "vehicles": len(self.tm.vehicles()) if not paused else 0,
                "label": self.last_event_label if now < self.hazard_until else ""}

    def stop(self):
        self.stop_ev.set()
        self.grabber.stop()


# ----------------------------------------------------------------------------- Engine
class Engine:
    def __init__(self, store: SettingsStore, on_message: Optional[Callable[[dict], None]] = None):
        self.store = store
        self.on_message = on_message or (lambda m: None)
        self.events = EventManager(store, lambda m: self.on_message(m))
        self.detector = None
        self.videomae = None
        self.load_errors: dict = {}
        self.pipelines: dict[str, CameraPipeline] = {}
        self._lock = threading.Lock()

    def load_models(self):
        s = self.store.settings
        pose = YOLO_POSE_WEIGHTS if Path(YOLO_POSE_WEIGHTS).exists() else None
        obj = YOLO_OBJ_WEIGHTS if Path(YOLO_OBJ_WEIGHTS).exists() else None
        if pose or obj:
            try:
                from app.analyzers.yolo_detector import YoloDetector
                self.detector = YoloDetector(pose, obj)
                print(f"[engine] YOLO sẵn sàng ({self.detector.device}); pose={bool(pose)} obj={bool(obj)}")
            except Exception as e:
                self.load_errors["yolo"] = f"{type(e).__name__}: {e}"
        else:
            self.load_errors["yolo"] = "không thấy yolo11n-pose.pt / yolo11n.pt trong models/"
        if Path(VIDEOMAE_WEIGHTS).exists() and Path(ACTION_VOCAB).exists():
            try:
                from app.analyzers.videomae import MultiTaskVideoMAEPipeline
                self.videomae = MultiTaskVideoMAEPipeline(VIDEOMAE_WEIGHTS, ACTION_VOCAB,
                                                          factory=s.videomae_factory)
                print(f"[engine] VideoMAE sẵn sàng ({self.videomae.device}, fp16={self.videomae.fp16})")
            except Exception as e:
                self.load_errors["videomae"] = f"{type(e).__name__}: {e}"
        else:
            self.load_errors["videomae"] = "không thấy best_multitask_videomae.pth / action_vocab.json"
        for k, v in self.load_errors.items():
            print(f"[engine] ⚠ {k}: {v}")

    def start(self):
        self.load_models()
        self.reload_cameras()

    def reload_cameras(self):
        with self._lock:
            wanted = {c.id: c for c in self.store.cameras if c.enabled}
            for cid in list(self.pipelines):
                p = self.pipelines[cid]
                c = wanted.get(cid)
                if c is None or (c.source, c.name, c.zone) != (p.cam.source, p.cam.name, p.cam.zone):
                    p.stop()
                    del self.pipelines[cid]
            for cid, c in wanted.items():
                if cid not in self.pipelines:
                    p = CameraPipeline(c, self)
                    self.pipelines[cid] = p
                    p.start()

    def stop(self):
        with self._lock:
            for p in self.pipelines.values():
                p.stop()
            self.pipelines.clear()

    def status(self) -> dict:
        return {"cameras": [p.status() for p in list(self.pipelines.values())],
                "modules": {"yolo_available": self.detector is not None,
                            "videomae_available": self.videomae is not None},
                "load_errors": self.load_errors, "ts": time.time()}
