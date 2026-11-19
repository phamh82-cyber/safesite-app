"""FastAPI server: REST API + MJPEG stream + WebSocket cảnh báo + Dashboard."""
from __future__ import annotations

import asyncio
import base64
import html
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import CLIPS_DIR, SNAPSHOTS_DIR, UPLOADS_DIR, CameraConfig, SettingsStore
from app.engine import Engine
from app.managers.event_manager import STATUSES

TEMPLATES = Path(__file__).parent / "templates"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
MAX_UPLOAD = 4 * 1024 ** 3   # 4 GB
STATUS_VN = {"new": "Chưa xem", "confirmed": "Đã xác nhận", "false_alarm": "Báo động giả"}


class CameraIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=500)
    zone: str = ""
    enabled: bool = True


class SourceTest(BaseModel):
    source: str


class SettingsIn(BaseModel):
    enable_yolo: Optional[bool] = None
    enable_videomae: Optional[bool] = None
    videomae_threshold: Optional[float] = Field(None, ge=0.05, le=0.99)
    lying_angle_deg: Optional[float] = Field(None, ge=30, le=85)
    safe_distance: Optional[float] = Field(None, ge=0, le=5)
    cooldown_s: Optional[float] = Field(None, ge=0, le=300)
    verify_interval_s: Optional[float] = Field(None, ge=0.2, le=10)
    yolo_conf: Optional[float] = Field(None, ge=0.05, le=0.95)
    fall_min_frames: Optional[int] = Field(None, ge=1, le=30)


class StatusIn(BaseModel):
    status: str


class Hub:
    """Phát tin nhắn từ thread của engine sang các WebSocket client."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def push(self, msg: dict) -> None:           # gọi được từ thread bất kỳ
        if self.loop and self.clients:
            asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)

    async def broadcast(self, msg: dict) -> None:
        data = json.dumps(msg, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                self.clients.discard(ws)


def create_app(store: SettingsStore) -> FastAPI:
    hub = Hub()
    engine = Engine(store, hub.push)

    async def status_loop():
        while True:
            await asyncio.sleep(1.0)
            if hub.clients:
                await hub.broadcast({"type": "status", **engine.status()})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(engine.start)          # nạp model không chặn event loop
        task = asyncio.create_task(status_loop())
        yield
        task.cancel()
        engine.stop()

    app = FastAPI(title="SafeSite", lifespan=lifespan)
    app.state.engine = engine
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")
    app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (TEMPLATES / "index.html").read_text(encoding="utf-8")

    # ---------------- stream, snapshot & websocket
    @app.get("/snapshot/{cam_id}")
    async def snapshot(cam_id: str):
        """Trả về 1 frame JPEG tĩnh của camera (dùng khi pause)."""
        pipe = engine.pipelines.get(cam_id)
        if pipe is None or not pipe.latest_jpeg:
            raise HTTPException(404, "camera không tồn tại hoặc chưa có frame")
        return Response(pipe.latest_jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/stream/{cam_id}")
    async def stream(cam_id: str):
        pipe = engine.pipelines.get(cam_id)
        if pipe is None:
            raise HTTPException(404, "camera không tồn tại hoặc đang tắt")

        async def gen():
            last = None
            while True:
                jpeg = pipe.latest_jpeg
                if jpeg and jpeg is not last:
                    last = jpeg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.03)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        hub.clients.add(websocket)
        try:
            await websocket.send_text(json.dumps({"type": "status", **engine.status()}))
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(websocket)

    # ---------------- cameras
    @app.get("/api/cameras")
    def cameras():
        st = {c["id"]: c for c in engine.status()["cameras"]}
        return [{**c.__dict__, "status": st.get(c.id)} for c in store.cameras]

    @app.post("/api/cameras")
    def add_camera(c: CameraIn):
        cam = CameraConfig(id="cam" + uuid.uuid4().hex[:6], **c.model_dump())
        store.set_cameras([*store.cameras, cam])
        engine.reload_cameras()
        return cam.__dict__

    @app.put("/api/cameras/{cam_id}")
    def edit_camera(cam_id: str, c: CameraIn):
        cams = [CameraConfig(id=cam_id, **c.model_dump()) if x.id == cam_id else x for x in store.cameras]
        if not any(x.id == cam_id for x in store.cameras):
            raise HTTPException(404)
        store.set_cameras(cams)
        engine.reload_cameras()
        return {"ok": True}

    @app.post("/api/upload")
    async def upload(request: Request, filename: str = "video.mp4", zone: str = ""):
        """Tải video thẳng lên (body = nội dung file thô) -> tự tạo 1 camera nguồn file và chạy phân tích."""
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"Định dạng không hỗ trợ: {ext or '(không có đuôi)'}")
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        clean = re.sub(r"[^\w.\-]", "_", Path(filename).name)[:80]
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{clean}"
        size = 0
        try:
            with open(dest, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "File quá lớn (tối đa 4 GB)")
                    await run_in_threadpool(f.write, chunk)
            ok = await run_in_threadpool(_readable, str(dest))
            if not ok:
                raise HTTPException(400, "Không đọc được video (file hỏng hoặc codec không hỗ trợ)")
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        cam = CameraConfig(id="cam" + uuid.uuid4().hex[:6], name=Path(filename).stem[:60],
                           source=str(dest), zone=zone or "Video tải lên")
        store.set_cameras([*store.cameras, cam])
        engine.reload_cameras()
        return cam.__dict__

    def _readable(path: str) -> bool:
        cap = cv2.VideoCapture(path)
        try:
            return cap.isOpened() and cap.read()[0]
        finally:
            cap.release()

    @app.delete("/api/cameras/{cam_id}")
    def del_camera(cam_id: str):
        for x in store.cameras:      # xoá luôn file video đã tải lên
            if x.id == cam_id and Path(x.source).parent == UPLOADS_DIR:
                engine.pipelines.get(cam_id) and engine.pipelines[cam_id].stop()
                time.sleep(0.3)
                Path(x.source).unlink(missing_ok=True)
        store.set_cameras([x for x in store.cameras if x.id != cam_id])
        engine.reload_cameras()
        return {"ok": True}

    @app.post("/api/cameras/{cam_id}/pause")
    async def pause_camera(cam_id: str, request: Request):
        pipe = engine.pipelines.get(cam_id)
        if pipe is None:
            raise HTTPException(404, "camera không tồn tại hoặc đang tắt")
        try:
            body = await request.json()
        except Exception:
            body = {}
        if body and "paused" in body:
            pipe.set_paused(bool(body["paused"]))
        else:
            pipe.toggle_paused()
        state = pipe.is_paused()
        hub.push({"type": "status", **engine.status()})
        return {"id": cam_id, "paused": state}

    @app.post("/api/cameras/test")
    def test_camera(t: SourceTest):
        src = t.source.strip()
        cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
        try:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
        finally:
            cap.release()
        if not ok:
            return {"ok": False, "message": "Không kết nối / không đọc được frame"}
        return {"ok": True, "message": f"OK {frame.shape[1]}x{frame.shape[0]}"}

    # ---------------- settings
    def settings_payload():
        from dataclasses import asdict
        st = engine.status()
        return {"settings": asdict(store.settings), "modules": st["modules"], "load_errors": st["load_errors"]}

    @app.get("/api/settings")
    def get_settings():
        return settings_payload()

    @app.put("/api/settings")
    def put_settings(p: SettingsIn):
        store.update_settings(p.model_dump(exclude_none=True))
        return settings_payload()

    # ---------------- events
    @app.get("/api/events")
    def events(date: str = "", status: str = "", camera_id: str = "", limit: int = 500):
        if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise HTTPException(400, "date phải dạng YYYY-MM-DD")
        return engine.events.db.list(date, status if status in STATUSES else "", camera_id, min(limit, 2000))

    @app.patch("/api/events/{event_id}")
    def set_status(event_id: int, b: StatusIn):
        if b.status not in STATUSES:
            raise HTTPException(400, "status không hợp lệ")
        if not engine.events.db.get(event_id):
            raise HTTPException(404)
        engine.events.db.update(event_id, status=b.status)
        return engine.events.db.get(event_id)

    @app.get("/api/events/{event_id}/report")
    def report(event_id: int):
        e = engine.events.db.get(event_id)
        if not e:
            raise HTTPException(404)
        img = ""
        if e["snapshot"] and (SNAPSHOTS_DIR / e["snapshot"]).exists():
            b64 = base64.b64encode((SNAPSHOTS_DIR / e["snapshot"]).read_bytes()).decode()
            img = f'<img style="max-width:100%" src="data:image/jpeg;base64,{b64}">'
        t = lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
        label = "hazard" if e["reason"] == "scan" else e["label"]
        rows = [("Mã sự cố", e["id"]), ("Loại sự cố", label), ("Nguyên nhân kích hoạt", e["reason"]),
                ("Camera", e["camera_name"]), ("Khu vực", e["zone"] or "-"),
                ("Bắt đầu", t(e["start_ts"])), ("Kết thúc", t(e["end_ts"])),
                ("Độ tin cậy", f'{e["confidence"] * 100:.1f}%'), ("Track ID", e["track_ids"] or "-"),
                ("Trạng thái", STATUS_VN.get(e["status"], e["status"]))]
        body = "".join(f"<tr><th align=left>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                       for k, v in rows)
        doc = (f"<!doctype html><meta charset=utf-8><title>Báo cáo sự cố #{e['id']}</title>"
               f"<body style='font-family:sans-serif;max-width:800px;margin:2em auto'>"
               f"<h2>SafeSite – Báo cáo sự cố #{e['id']}</h2><table cellpadding=6 border=1 "
               f"style='border-collapse:collapse'>{body}</table><h3>Ảnh minh chứng</h3>{img}</body>")
        return Response(doc, media_type="text/html; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="incident_{e["id"]}.html"'})

    return app
