"""Lưu trữ sự kiện (SQLite), ảnh chụp, video ngắn trước/sau sự kiện và phát realtime."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import CLIP_DIR, DB_PATH, SNAP_DIR

EVENT_LABELS = {
    "FALL": "Té ngã",
    "FALL_IMMOBILE": "Bất động sau ngã",
    "PERSON_DOWN": "Người nằm bất động",
    "NEAR_MISS": "Nguy cơ va chạm người – xe",
    "PERSON_VEHICLE_COLLISION": "Va chạm người – xe",
    "VEHICLE_COLLISION": "Va chạm xe – xe",
    "FALL_FROM_HEIGHT": "Ngã từ trên cao",
    "OBJECT_FALL": "Vật rơi",
    "STRUCK_BY": "Bị vật rơi đập trúng",
    "FIRE_SMOKE": "Cháy / khói",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ev_key TEXT, camera_id TEXT, camera_name TEXT, type TEXT, severity TEXT,
  confidence REAL, t_video REAL, created_at TEXT, track_ids TEXT, details TEXT,
  snapshot TEXT, clip TEXT, status TEXT DEFAULT 'new', note TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
"""


class EventStore:
    def __init__(self, path: Path = DB_PATH):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def insert(self, cam_id: str, cam_name: str, ev: dict, snapshot: str, clip: str) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO events(ev_key,camera_id,camera_name,type,severity,confidence,t_video,"
                "created_at,track_ids,details,snapshot,clip) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (ev["key"], cam_id, cam_name, ev["type"], ev["severity"], ev["confidence"], ev["t"],
                 datetime.now().isoformat(timespec="seconds"), json.dumps(ev["track_ids"]),
                 json.dumps(ev.get("details", {}), ensure_ascii=False), snapshot, clip))
            self.conn.commit()
            return int(cur.lastrowid)

    def merge_details(self, ev_key: str, patch: dict) -> Optional[int]:
        with self.lock:
            row = self.conn.execute("SELECT id, details FROM events WHERE ev_key=?", (ev_key,)).fetchone()
            if not row:
                return None
            d = json.loads(row["details"] or "{}")
            d.update(patch)
            self.conn.execute("UPDATE events SET details=? WHERE id=?",
                              (json.dumps(d, ensure_ascii=False), row["id"]))
            self.conn.commit()
            return int(row["id"])

    def set_clip(self, ev_id: int, clip: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE events SET clip=? WHERE id=?", (clip, ev_id))
            self.conn.commit()

    def update(self, ev_id: int, status: Optional[str], note: Optional[str]) -> None:
        with self.lock:
            if status is not None:
                self.conn.execute("UPDATE events SET status=? WHERE id=?", (status, ev_id))
            if note is not None:
                self.conn.execute("UPDATE events SET note=? WHERE id=?", (note, ev_id))
            self.conn.commit()

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["track_ids"] = json.loads(d["track_ids"] or "[]")
        d["details"] = json.loads(d["details"] or "{}")
        d["label"] = EVENT_LABELS.get(d["type"], d["type"])
        return d

    def get(self, ev_id: int) -> Optional[dict]:
        with self.lock:
            r = self.conn.execute("SELECT * FROM events WHERE id=?", (ev_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, camera: str = "", typ: str = "", severity: str = "", status: str = "",
             limit: int = 50, offset: int = 0) -> dict:
        where, args = [], []
        for col, val in (("camera_id", camera), ("type", typ), ("severity", severity),
                         ("status", status)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        w = ("WHERE " + " AND ".join(where)) if where else ""
        with self.lock:
            total = self.conn.execute(f"SELECT COUNT(*) FROM events {w}", args).fetchone()[0]
            rows = self.conn.execute(f"SELECT * FROM events {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                                     args + [limit, offset]).fetchall()
        return {"total": total, "items": [self._row(r) for r in rows]}

    def stats(self) -> dict:
        with self.lock:
            by_type = dict(self.conn.execute("SELECT type, COUNT(*) FROM events GROUP BY type").fetchall())
            by_status = dict(self.conn.execute(
                "SELECT status, COUNT(*) FROM events GROUP BY status").fetchall())
            by_severity = dict(self.conn.execute(
                "SELECT severity, COUNT(*) FROM events GROUP BY severity").fetchall())
            by_day = self.conn.execute(
                "SELECT substr(created_at,1,10) d, type, COUNT(*) FROM events "
                "GROUP BY d, type ORDER BY d DESC LIMIT 200").fetchall()
            by_hour = self.conn.execute(
                "SELECT CAST(substr(created_at,12,2) AS INTEGER) h, COUNT(*) FROM events "
                "GROUP BY h").fetchall()
        return {"by_type": by_type, "by_status": by_status, "by_severity": by_severity,
                "by_day": [{"day": d, "type": t, "count": c} for d, t, c in by_day],
                "by_hour": {int(h): c for h, c in by_hour}, "labels": EVENT_LABELS}

    def clear(self) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM events")
            self.conn.commit()


class ClipRecorder:
    """Giữ vòng đệm các khung hình đã vẽ; khi có sự kiện ghi `pre` giây trước và `post`
    giây sau thành file .webm (VP8 – trình duyệt phát được)."""

    def __init__(self, pre: float, post: float):
        self.pre, self.post = pre, post
        self.buf: deque = deque()
        self.pending: list = []
        self.lock = threading.Lock()

    def push(self, t: float, frame: np.ndarray) -> None:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        with self.lock:
            if self.buf and t < self.buf[-1][0]:   # video lặp: đồng hồ quay về 0 ⇒ bỏ đệm cũ
                self.buf.clear()
            self.buf.append((t, jpg))
            while self.buf and self.buf[0][0] < t - self.pre:
                self.buf.popleft()
            done = []
            for p in self.pending:
                p["frames"].append((t, jpg))
                if t >= p["end"]:
                    done.append(p)
            for p in done:
                self.pending.remove(p)
        for p in done:
            threading.Thread(target=self._write, args=(p,), daemon=True).start()

    def flush(self) -> None:
        """Camera dừng / hết video: ghi nốt các clip còn chờ `post` giây (có bao nhiêu ghi bấy nhiêu)."""
        with self.lock:
            pend, self.pending = self.pending, []
        for p in pend:
            threading.Thread(target=self._write, args=(p,), daemon=True).start()

    def request(self, t: float, path: Path, on_done: Callable[[str], None]) -> None:
        with self.lock:
            self.pending.append({"end": t + self.post, "frames": list(self.buf), "path": path,
                                 "cb": on_done})

    @staticmethod
    def _write(p: dict) -> None:
        frames = p["frames"]
        if len(frames) < 2:
            return
        dur = frames[-1][0] - frames[0][0]
        fps = max(2.0, min(30.0, (len(frames) - 1) / max(dur, 1e-3)))
        first = cv2.imdecode(frames[0][1], cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        path: Path = p["path"]
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
        if not wr.isOpened():
            path = path.with_suffix(".mp4")
            wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _, jpg in frames:
            wr.write(cv2.imdecode(jpg, cv2.IMREAD_COLOR))
        wr.release()
        p["cb"](path.name)


class EventHub:
    """Điểm trung tâm: worker camera gọi `emit`, hub lưu và đẩy qua WebSocket."""

    def __init__(self, store: EventStore):
        self.store = store
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: set = set()
        self._recent: deque = deque(maxlen=200)   # cho banner trên khung hình

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def _broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def broadcast(self, msg: dict) -> None:
        if self.loop and self.clients:
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self.loop)

    def emit(self, cam_id: str, cam_name: str, ev: dict, frame: np.ndarray,
             clips: Optional[ClipRecorder]) -> None:
        if "update_of" in ev:
            if ev["update_of"]:
                eid = self.store.merge_details(ev["update_of"], ev.get("details", {}))
                if eid:
                    self.broadcast({"kind": "event_update", "event": self.store.get(eid)})
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap = f"{cam_id}_{stamp}_{ev['type']}.jpg"
        img = frame.copy()
        for b in ev.get("boxes", []):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imwrite(str(SNAP_DIR / snap), img)
        eid = self.store.insert(cam_id, cam_name, ev, snap, "")
        self._recent.append((time.time(), cam_id, ev["type"]))
        if clips is not None:
            def done(name: str, eid=eid):
                self.store.set_clip(eid, name)
                self.broadcast({"kind": "event_update", "event": self.store.get(eid)})
            clips.request(ev["t"], CLIP_DIR / f"{cam_id}_{stamp}_{ev['type']}.webm", done)
        self.broadcast({"kind": "event", "event": self.store.get(eid)})

    def recent_for(self, cam_id: str, within: float = 4.0) -> list:
        now = time.time()
        return [typ for ts, c, typ in self._recent if c == cam_id and now - ts <= within]
