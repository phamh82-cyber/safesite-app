"""Đọc video/camera → YOLO (pose cho người, detect cho xe) + ByteTrack → Engine → vẽ,
lưu sự kiện. Mỗi camera chạy trong một luồng riêng."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from .annotate import draw
from .config import DATA_DIR, ROOT, SETTINGS, VEHICLE_CLASSES, Settings
from .engine import Engine
from .events import ClipRecorder, EventHub
from .tracks import Detection

log = logging.getLogger("safesite")
TRACKER_YAML = DATA_DIR / "bytetrack_safety.yaml"


def ensure_tracker_cfg(process_fps: float) -> str:
    # track_buffer tính theo khung hình đã phân tích: giữ track mất dấu ~3 giây
    TRACKER_YAML.write_text(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.25\n"
        f"track_buffer: {max(10, int(3 * process_fps))}\nmatch_thresh: 0.8\nfuse_score: True\n",
        encoding="utf-8")
    return str(TRACKER_YAML)


def fire_kind(name: str) -> Optional[str]:
    """Tên lớp của mô hình cháy/khói → 'fire' | 'smoke' | None."""
    n = str(name).lower()
    if "smoke" in n:
        return "smoke"
    if "fire" in n or "flame" in n:
        return "fire"
    return None


class Detector:
    """Mô hình YOLO: pose (người + 17 keypoint), detect (phương tiện) và tùy chọn cháy/khói."""

    def __init__(self, s: Settings):
        from ultralytics import YOLO   # import muộn để kiểm thử logic không cần torch
        self.s = s
        self.pose = YOLO(s.pipeline.pose_model)
        self.det = YOLO(s.pipeline.det_model)
        self.tracker = ensure_tracker_cfg(s.pipeline.process_fps)
        self.fire = None
        self.fire_map: dict = {}
        h = s.hazard
        if h.enabled and h.fire_model:
            path = Path(h.fire_model)
            path = path if path.is_absolute() else ROOT / path
            if not path.is_file():
                log.warning("Không thấy mô hình cháy/khói %s – dùng bộ lọc màu thay thế", path)
            else:
                m = YOLO(str(path))
                fmap = {int(i): fire_kind(n) for i, n in dict(m.names).items() if fire_kind(n)}
                if not fmap:
                    log.warning("Mô hình %s không có lớp fire/smoke (lớp: %s) – bỏ qua",
                                path.name, list(dict(m.names).values()))
                else:
                    self.fire, self.fire_map = m, fmap

    def detect_fire(self, frame: np.ndarray) -> list:
        """[(kind, conf, box_xyxy)] của khung này."""
        h, p = self.s.hazard, self.s.pipeline
        r = self.fire.predict(frame, conf=h.conf, imgsz=p.imgsz, verbose=False)[0]
        out: list = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(cls)):
            kind = self.fire_map.get(int(cls[i]))
            if kind:
                out.append((kind, float(confs[i]), boxes[i]))
        return out

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        p = self.s.pipeline
        out: List[Detection] = []
        r = self.pose.track(frame, persist=True, tracker=self.tracker, conf=p.person_conf,
                            imgsz=p.imgsz, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for i in range(len(ids)):
                out.append(Detection(int(ids[i]), "person", "person", boxes[i], float(confs[i]),
                                     kp[i] if kp is not None else None))
        r = self.det.track(frame, persist=True, tracker=self.tracker, conf=p.vehicle_conf,
                           imgsz=p.imgsz, classes=list(VEHICLE_CLASSES), verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(ids)):
                out.append(Detection(int(ids[i]), "vehicle", VEHICLE_CLASSES.get(int(cls[i]), "vehicle"),
                                     boxes[i], float(confs[i])))
        return out


def is_network(source) -> bool:
    return isinstance(source, str) and "://" in source


def open_source(source) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        return cv2.VideoCapture(int(source))
    if is_network(source):
        # camera mạng: không chờ quá 5 s khi mở/đọc (mặc định FFmpeg chờ 30 s)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)


class LiveReader:
    """Đọc camera trực tiếp trong luồng riêng và chỉ giữ khung mới nhất: khi phân tích
    chậm hơn tốc độ camera, khung cũ bị bỏ thay vì xếp hàng gây trễ dồn.
    Camera mạng (http/rtsp) tự kết nối lại khi rớt."""

    retry_delay = 1.0             # giây chờ giữa hai lần kết nối lại

    def __init__(self, source):
        self.source = source
        self.cap = open_source(source)
        self.cond = threading.Condition()
        self.frame: Optional[np.ndarray] = None
        self.t = 0.0              # thời điểm nhận khung (time.monotonic)
        self.seq = 0
        self.dead = False         # hết hình và không kết nối lại được (webcam USB bị rút)
        self._stop = threading.Event()
        # camera mạng chưa mở được (điện thoại chưa phát hình) vẫn chạy luồng để thử lại
        if self.isOpened():
            threading.Thread(target=self._loop, daemon=True).start()

    def isOpened(self) -> bool:
        return is_network(self.source) or self.cap.isOpened()

    def get(self, prop) -> float:
        return self.cap.get(prop)

    def _loop(self) -> None:
        cap = self.cap
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok:
                with self.cond:
                    self.frame, self.t, self.seq = frame, time.monotonic(), self.seq + 1
                    self.cond.notify_all()
                continue
            if not is_network(self.source):
                break
            cap.release()
            self._stop.wait(self.retry_delay)
            cap = open_source(self.source)
        cap.release()
        with self.cond:
            self.dead = True
            self.cond.notify_all()

    def read(self, last_seq: int, timeout: float = 0.5):
        """Chờ khung mới hơn last_seq. Trả về (seq, frame, t); frame = None nếu hết giờ chờ."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > last_seq or self.dead, timeout)
            if self.seq > last_seq:
                return self.seq, self.frame, self.t
            return last_seq, None, 0.0

    def release(self) -> None:
        self._stop.set()



class CameraWorker(threading.Thread):
    restart_backoff = 2.0          # giây chờ trước khi tự chạy lại sau lỗi (nhân đôi tới max_backoff)
    max_backoff = 30.0
    max_frame_errors = 3           # số khung liên tiếp lỗi thì dựng lại toàn bộ pipeline

    def __init__(self, cam: dict, hub: EventHub, settings: Settings = SETTINGS):
        super().__init__(daemon=True)
        self.cam, self.hub, self.s = cam, hub, settings
        self.stop_flag = threading.Event()
        self.frame_lock = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.raw: Optional[np.ndarray] = None
        self.status = "starting"
        self.error = ""
        self.fps = 0.0
        self.progress = 0.0
        self.engine: Optional[Engine] = None
        self.counts = {"persons": 0, "vehicles": 0}
        self.restarts = 0

    # ------------------------------------------------------------------
    def info(self) -> dict:
        d = {"status": self.status, "error": self.error, "fps": round(self.fps, 1),
             "progress": round(self.progress, 3), **self.counts}
        d["restarts"] = self.restarts
        eng = self.engine
        if eng:
            d["ground"] = eng.mapper.info()
            d["shaking"] = bool(eng.shaking)
        return d

    def apply_calibration(self) -> None:
        if self.engine is None:
            return
        cal = self.cam.get("calibration") or {}
        self.engine.set_calibration(cal.get("img_pts"), cal.get("ground_pts"))

    def _publish(self, img: np.ndarray) -> None:
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.frame_lock:
                self.jpeg = jpg.tobytes()
                self.frame_lock.notify_all()

    def run(self) -> None:
        """Giám sát: lỗi bất ngờ không giết camera mà ghi log đầy đủ rồi tự chạy lại."""
        backoff = self.restart_backoff
        while not self.stop_flag.is_set():
            started = time.monotonic()
            try:
                self._run()
                return
            except Exception as e:
                log.exception("Camera %s gặp lỗi", self.cam.get("name"))
                if time.monotonic() - started > 60:
                    backoff = self.restart_backoff           # đã chạy ổn định một lúc rồi mới lỗi
                self.restarts += 1
                self.status = "error"
                self.error = f"{type(e).__name__}: {e} – tự chạy lại sau {backoff:.0f} s"
                if self.stop_flag.wait(backoff):
                    return
                backoff = min(backoff * 2, self.max_backoff)
                self.status = "starting"

    def _run(self) -> None:
        src = self.cam["source"]
        is_file = self.cam.get("source_type") == "file"
        cap = open_source(src) if is_file else LiveReader(src)
        if not cap.isOpened():
            cap.release()
            self.status, self.error = "error", f"Không mở được nguồn video: {src}"
            return
        self.engine = None
        self.error = ""
        clips = ClipRecorder(self.s.pipeline.clip_pre, self.s.pipeline.clip_post)
        try:
            self._loop(cap, src, is_file, clips)
        finally:
            cap.release()
            clips.flush()

    def _loop(self, cap, src, is_file: bool, clips: ClipRecorder) -> None:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if src_fps <= 1 or src_fps > 120:
            src_fps = 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        detector = Detector(self.s)
        self.status = "running"
        idx, wall0, last_t, t_hist = 0, time.monotonic(), -1e9, []
        seq, last_frame = 0, time.monotonic()
        n_proc, err_streak = 0, 0
        while not self.stop_flag.is_set():
            if is_file:
                step = max(1, int(round(src_fps / self.s.pipeline.process_fps)))
                ok, frame = cap.read()
                if not ok:
                    if self.cam.get("loop"):
                        cap.release()
                        cap = open_source(src)
                        idx, wall0 = 0, time.monotonic()
                        detector = Detector(self.s)
                        self.engine = None
                        continue
                    self.status = "finished"
                    break
                idx += 1
                if (idx - 1) % step:
                    continue
                t = (idx - 1) / src_fps
                if n_frames:
                    self.progress = idx / n_frames
                if self.s.pipeline.realtime_files:
                    lag = t - (time.monotonic() - wall0)
                    if lag > 0:
                        time.sleep(lag)
            else:
                seq, frame, t_cap = cap.read(seq)
                if frame is None:
                    if cap.dead:
                        self.status, self.error = "error", "Mất kết nối camera"
                        break
                    if time.monotonic() - last_frame > 3.0:
                        self.status, self.error = "reconnecting", "Không nhận được hình – đang kết nối lại"
                    continue
                last_frame = time.monotonic()
                if self.status == "reconnecting":
                    self.status, self.error = "running", ""
                t = t_cap - wall0     # dùng thời điểm nhận khung, không phải lúc xử lý xong
                if t - last_t < 1.0 / self.s.pipeline.process_fps:
                    continue
            last_t = t
            try:
                if self.engine is None:
                    h, w = frame.shape[:2]
                    self.engine = Engine(self.s, w, h)
                    self.engine.hazard.model_active = detector.fire is not None
                    self.apply_calibration()
                self.raw = frame
                dets = detector(frame)
                hazards = None
                if detector.fire is not None and n_proc % max(1, self.s.hazard.every) == 0:
                    hazards = detector.detect_fire(frame)
                n_proc += 1
                events = self.engine.process(t, dets, frame, hazards)
                self.counts = {"persons": sum(1 for x in self.engine.tm.persons() if not x.lost),
                               "vehicles": sum(1 for x in self.engine.tm.vehicles() if not x.lost)}
                img = draw(frame, self.engine, self.hub.recent_for(self.cam["id"]) +
                           [e["type"] for e in events if "type" in e], self.s.pipeline.show_bev)
                for ev in events:
                    self.hub.emit(self.cam["id"], self.cam["name"], ev, img, clips)
                clips.push(t, img)
                self._publish(img)
                err_streak = 0
            except Exception:
                err_streak += 1
                log.exception("Camera %s: lỗi xử lý khung hình (%d liên tiếp)",
                              self.cam.get("name"), err_streak)
                if err_streak >= self.max_frame_errors:
                    raise            # run() sẽ ghi log và dựng lại pipeline từ đầu
                continue
            now = time.monotonic()
            t_hist = [x for x in t_hist if now - x < 2.0] + [now]
            self.fps = (len(t_hist) - 1) / max(t_hist[-1] - t_hist[0], 1e-3) if len(t_hist) > 1 else 0
        if self.status == "running":
            self.status = "stopped"

    def stop(self) -> None:
        self.stop_flag.set()


# ---------------------------------------------------------------------------
def process_video(path: str, settings: Settings, on_frame: Optional[Callable] = None,
                  writer_path: Optional[str] = None) -> List[dict]:
    """Chạy không giao diện (dùng cho tools/evaluate.py và tools/run_video.py)."""
    cap = open_source(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / settings.pipeline.process_fps)))
    det = Detector(settings)
    eng, wr, events, idx, banner, n_proc = None, None, [], 0, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % step:
            continue
        t = (idx - 1) / src_fps
        if eng is None:
            h, w = frame.shape[:2]
            eng = Engine(settings, w, h)
            eng.hazard.model_active = det.fire is not None
        hz = det.detect_fire(frame) if det.fire is not None and n_proc % max(1, settings.hazard.every) == 0 else None
        n_proc += 1
        evs = [e for e in eng.process(t, det(frame), frame, hz) if "type" in e]
        events += evs
        if writer_path or on_frame:
            banner = [(tt, ty) for tt, ty in banner if t - tt <= 3.0] + [(t, e["type"]) for e in evs]
            img = draw(frame, eng, [ty for _, ty in banner], settings.pipeline.show_bev)
            if writer_path:
                if wr is None:
                    fps_out = src_fps / step
                    wr = cv2.VideoWriter(writer_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out,
                                         (img.shape[1], img.shape[0]))
                wr.write(img)
            if on_frame:
                on_frame(t, img, evs)
    cap.release()
    if wr:
        wr.release()
    return events
