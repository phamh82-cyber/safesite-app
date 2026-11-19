"""Event Manager: Fusion (VideoMAE + hình học YOLO + tracking), Cooldown, lưu minh chứng, nhật ký SQLite."""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import CLIPS_DIR, DB_PATH, SNAPSHOTS_DIR, CameraConfig

STATUSES = ("new", "confirmed", "false_alarm")

GEOM_LABEL = {"fall": "person falling down", "vehicle_proximity": "person too close to vehicle",
              "fire": "fire detected", "scan": "hazard"}


# ----------------------------------------------------------------------------- DB
class EventStore:
    def __init__(self, path=DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("""CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT, camera_name TEXT, zone TEXT,
                label TEXT, confidence REAL, reason TEXT, track_ids TEXT,
                start_ts REAL, end_ts REAL, status TEXT DEFAULT 'new', clip TEXT, snapshot TEXT)""")
            self._db.commit()

    def insert(self, **kw) -> int:
        cols = ",".join(kw)
        with self._lock:
            cur = self._db.execute(f"INSERT INTO events({cols}) VALUES({','.join('?' * len(kw))})",
                                   list(kw.values()))
            self._db.commit()
            return cur.lastrowid

    def update(self, event_id: int, **kw) -> None:
        sets = ",".join(f"{k}=?" for k in kw)
        with self._lock:
            self._db.execute(f"UPDATE events SET {sets} WHERE id=?", [*kw.values(), event_id])
            self._db.commit()

    def get(self, event_id: int) -> Optional[dict]:
        with self._lock:
            r = self._db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, date: str = "", status: str = "", camera_id: str = "", limit: int = 500):
        q, args = "SELECT * FROM events WHERE 1=1", []
        if date:
            d0 = datetime.strptime(date, "%Y-%m-%d").timestamp()
            q += " AND start_ts>=? AND start_ts<?"
            args += [d0, d0 + 86400]
        if status:
            q += " AND status=?"
            args.append(status)
        if camera_id:
            q += " AND camera_id=?"
            args.append(camera_id)
        q += " ORDER BY start_ts DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["clip_url"] = f"/clips/{d['clip']}" if d.get("clip") else None
        d["snapshot_url"] = f"/snapshots/{d['snapshot']}" if d.get("snapshot") else None
        return d


# ----------------------------------------------------------------------------- clip writer
def write_clip(jpegs: list, fps: float, path: Path) -> bool:
    """Ghi list JPEG -> mp4 (H.264 để trình duyệt phát được). Ưu tiên avc1; nếu không có thì mp4v + ffmpeg."""
    if not jpegs:
        return False
    first = cv2.imdecode(np.frombuffer(jpegs[0], np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return False
    h, w = first.shape[:2]
    fps = float(min(max(fps, 5.0), 60.0))
    path = Path(path)

    def _write(fourcc, target):
        vw = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not vw.isOpened():
            return False
        for j in jpegs:
            f = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
            if f is not None:
                vw.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
        vw.release()
        return True

    if _write("avc1", path) and path.stat().st_size > 1000:
        return True
    tmp = path.with_suffix(".tmp.mp4")
    if not _write("mp4v", tmp):
        return False
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)])
        tmp.unlink(missing_ok=True)
        return r.returncode == 0 and path.exists()
    tmp.replace(path)     # không có ffmpeg: file vẫn tải về được nhưng trình duyệt có thể không phát
    return True


# ----------------------------------------------------------------------------- manager
class EventManager:
    def __init__(self, store, on_event: Optional[Callable[[dict], None]] = None):
        self.store = store                # SettingsStore
        self.db = EventStore()
        self.on_event = on_event or (lambda msg: None)
        self._last: dict = {}             # (camera_id, key) -> ts
        self._pending: dict = {}          # camera_id -> list[dict] (clip đang chờ frame "sau sự cố")
        self._lock = threading.Lock()

    def in_cooldown(self, camera_id: str, key: str, ts: float) -> bool:
        with self._lock:
            last = self._last.get((camera_id, key))
        return last is not None and ts - last < self.store.settings.cooldown_s

    # ---------------- Fusion
    def fuse(self, cand, vm: Optional[dict], s):
        """Trả về (confirmed, label, confidence)."""
        geom_label = GEOM_LABEL.get(cand.reason, cand.reason)
        if vm is None:                                # VideoMAE tắt -> chỉ dựa hình học
            ok = cand.score >= 0.7 and (cand.reason != "vehicle_proximity" or cand.score >= 0.8)
            label = "hazard" if cand.reason == "scan" else geom_label
            return ok, label, cand.score
        p = vm["hazard_prob"]
        agree = cand.reason != "scan" and cand.score >= 0.7
        ok = p >= s.videomae_threshold or (agree and p >= s.videomae_threshold * s.consensus_ratio)
        label = "hazard" if cand.reason == "scan" else (vm.get("top_action") or geom_label)
        return ok, label, p

    # ---------------- Xử lý kết quả xác thực
    def handle(self, cam: CameraConfig, cand, vm: Optional[dict], frame: Optional[np.ndarray],
               ts: float, pre_frames: Optional[list] = None) -> Optional[dict]:
        """pre_frames: list (ts, jpeg) các frame TRƯỚC sự cố lấy từ EvidenceBuffer."""
        s = self.store.settings
        key = (cam.id, cand.key)
        with self._lock:
            last = self._last.get(key)
            if last is not None and ts - last < s.cooldown_s:
                return None                           # chống spam
            ok, label, conf = self.fuse(cand, vm, s)
            if not ok:
                return None
            self._last[key] = ts

        stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
        base = f"{cam.id}_{stamp}_{cand.reason}"
        snap_name = f"{base}.jpg"
        if frame is not None:
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(SNAPSHOTS_DIR / snap_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            snap_name = None
        eid = self.db.insert(camera_id=cam.id, camera_name=cam.name, zone=cam.zone, label=label,
                             confidence=float(conf), reason=cand.reason,
                             track_ids=",".join(map(str, cand.track_ids)), start_ts=ts,
                             end_ts=ts + s.clip_post_s, status="new", clip=None, snapshot=snap_name)
        with self._lock:
            self._pending.setdefault(cam.id, []).append(
                {"id": eid, "name": f"{base}.mp4", "frames": list(pre_frames or []),
                 "deadline": ts + s.clip_post_s})
        ev = self.db.get(eid)
        self.on_event({"type": "event", "event": ev})
        return ev

    # ---------------- Nhận frame (đã encode JPEG) sau sự cố để hoàn tất clip
    def feed_frame(self, camera_id: str, ts: float, jpeg: bytes) -> None:
        done = []
        with self._lock:
            pend = self._pending.get(camera_id)
            if not pend:
                return
            for p in pend:
                p["frames"].append((ts, jpeg))
                if ts >= p["deadline"]:
                    done.append(p)
            self._pending[camera_id] = [p for p in pend if p not in done]
        for p in done:
            threading.Thread(target=self._finish, args=(p,), daemon=True).start()

    def _finish(self, p: dict) -> None:
        frames = sorted(dict(p["frames"]).items())      # khử trùng theo timestamp
        if len(frames) < 2:
            return
        span = frames[-1][0] - frames[0][0]
        fps = (len(frames) - 1) / span if span > 0 else 25.0
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if write_clip([j for _, j in frames], fps, CLIPS_DIR / p["name"]):
                self.db.update(p["id"], clip=p["name"], end_ts=frames[-1][0])
                self.on_event({"type": "event_update", "event": self.db.get(p["id"])})
        except Exception as e:  # pragma: no cover
            print(f"[event] ghi clip lỗi: {e}")
