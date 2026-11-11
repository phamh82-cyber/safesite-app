"""Web server: REST API + WebSocket + MJPEG stream + giao diện dashboard."""
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import CAMERAS_PATH, CLIP_DIR, ROOT, SETTINGS, SNAP_DIR, UPLOAD_DIR, load_settings, save_settings
from .events import EventHub, EventStore
from .pipeline import CameraWorker

app = FastAPI(title="SafeSite – Giám sát tai nạn lao động")
store = EventStore()
hub = EventHub(store)
cameras: dict[str, dict] = {}
workers: dict[str, CameraWorker] = {}


# ----------------------------------------------------------------- lưu cấu hình camera
def save_cameras() -> None:
    CAMERAS_PATH.write_text(json.dumps(list(cameras.values()), indent=2, ensure_ascii=False),
                            encoding="utf-8")


def load_cameras() -> None:
    if CAMERAS_PATH.exists():
        for c in json.loads(CAMERAS_PATH.read_text(encoding="utf-8")):
            cameras[c["id"]] = c


SOURCE_TYPES = {"file", "webcam", "rtsp"}
NET_SCHEMES = {"rtsp", "rtsps", "http", "https", "rtmp"}


def mask_source(src: str) -> str:
    """rtsp://user:matkhau@host → rtsp://user:***@host (không trả mật khẩu về trình duyệt)."""
    if "://" not in src:
        return src
    p = urlsplit(src)
    if p.password is None:
        return src
    userinfo, _, host = p.netloc.rpartition("@")
    return urlunsplit(p._replace(netloc=f"{userinfo.split(':', 1)[0]}:***@{host}"))


def restore_password(new: str, old: str) -> str:
    """Form sửa camera gửi lại địa chỉ đã che (user:***@host): khôi phục mật khẩu cũ nếu cùng user."""
    if ":***@" not in new or "://" not in old:
        return new
    po, pn = urlsplit(old), urlsplit(new)
    userinfo, _, host = pn.netloc.rpartition("@")
    user = userinfo.split(":", 1)[0]
    if po.password is None or user != po.username:
        return new
    return urlunsplit(pn._replace(netloc=f"{user}:{po.password}@{host}"))


def validate_source(stype: str, source: str) -> None:
    """Dùng chung cho thêm (POST) và sửa (PUT) camera."""
    if stype not in SOURCE_TYPES:
        raise HTTPException(400, "Loại nguồn không hợp lệ")
    if stype == "file":
        p = Path(source).resolve() if source else None
        if p is None or not p.is_file():
            raise HTTPException(400, "Không tìm thấy file video")
        if UPLOAD_DIR.resolve() not in p.parents:
            raise HTTPException(400, "Chỉ nhận file video đã tải lên (thư mục data/uploads)")
    elif stype == "webcam":
        if not source.isdigit():
            raise HTTPException(400, "Webcam cần số thứ tự, VD: 0")
    else:
        u = urlsplit(source)
        if u.scheme.lower() not in NET_SCHEMES or not u.netloc:
            raise HTTPException(400, "Địa chỉ camera phải bắt đầu bằng rtsp://, http:// hoặc https://")


def cam_view(c: dict) -> dict:
    w = workers.get(c["id"])
    d = dict(c)
    d["source"] = mask_source(d.get("source", ""))
    d["runtime"] = w.info() if w and w.is_alive() else (w.info() if w else {"status": "stopped"})
    return d


def start_worker(cid: str) -> None:
    old = workers.get(cid)
    if old and old.is_alive():
        return
    w = CameraWorker(cameras[cid], hub, SETTINGS)
    workers[cid] = w
    w.start()


def restart_worker(cid: str) -> None:
    old = workers.pop(cid, None)
    if old:
        old.stop()
        old.join(timeout=5.0)   # luồng cũ dừng sau khung đang xử lý
    start_worker(cid)


@app.on_event("startup")
async def _startup() -> None:
    load_settings()
    load_cameras()
    hub.attach_loop(asyncio.get_running_loop())
    for cid, c in cameras.items():
        if c.get("autostart"):
            start_worker(cid)
    asyncio.create_task(_status_loop())


async def _status_loop() -> None:
    while True:
        await asyncio.sleep(1.0)
        if hub.clients:
            await hub._broadcast({"kind": "status", "cameras": [cam_view(c) for c in cameras.values()]})


# ----------------------------------------------------------------- camera
class CameraIn(BaseModel):
    name: str
    source_type: str          # file | webcam | rtsp
    source: str
    loop: bool = False
    autostart: bool = False


@app.get("/api/cameras")
def list_cameras():
    return [cam_view(c) for c in cameras.values()]


@app.post("/api/cameras")
def add_camera(body: CameraIn):
    data = body.model_dump()
    data["name"], data["source"] = data["name"].strip(), data["source"].strip()
    if not data["name"] or not data["source"]:
        raise HTTPException(400, "Tên và địa chỉ nguồn không được để trống")
    validate_source(data["source_type"], data["source"])
    cid = uuid.uuid4().hex[:8]
    cameras[cid] = {"id": cid, **data, "calibration": None}
    save_cameras()
    start_worker(cid)
    return cam_view(cameras[cid])


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(400, "Định dạng video không hỗ trợ")
    dst = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}{suffix}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dst), "name": file.filename}


def _cam(cid: str) -> dict:
    if cid not in cameras:
        raise HTTPException(404, "Không có camera")
    return cameras[cid]


@app.post("/api/cameras/{cid}/start")
def cam_start(cid: str):
    _cam(cid)
    start_worker(cid)
    return cam_view(cameras[cid])


@app.post("/api/cameras/{cid}/stop")
def cam_stop(cid: str):
    _cam(cid)
    w = workers.get(cid)
    if w:
        w.stop()
    return {"ok": True}


class CameraPatch(BaseModel):
    name: Optional[str] = None
    source_type: Optional[str] = None
    source: Optional[str] = None
    loop: Optional[bool] = None
    autostart: Optional[bool] = None


@app.put("/api/cameras/{cid}")
def cam_update(cid: str, body: CameraPatch):
    """Sửa camera, giữ nguyên ID và hiệu chỉnh. Đổi nguồn (VD: IP điện thoại mới) thì
    camera được chạy lại với nguồn mới."""
    c = _cam(cid)
    data = body.model_dump(exclude_none=True)
    for k in ("name", "source"):
        if k in data:
            data[k] = data[k].strip()
            if not data[k]:
                raise HTTPException(400, "Tên và địa chỉ nguồn không được để trống")
    if "source" in data:
        data["source"] = restore_password(data["source"], c["source"])
    stype = data.get("source_type", c["source_type"])
    source = data.get("source", c["source"])
    if stype not in SOURCE_TYPES:
        raise HTTPException(400, "Loại nguồn không hợp lệ")
    if (stype, source) != (c["source_type"], c["source"]):   # chỉ kiểm tra lại khi đổi nguồn
        validate_source(stype, source)
    changed = (stype, source) != (c["source_type"], c["source"])
    c.update(data)
    save_cameras()
    if changed:
        restart_worker(cid)
    return cam_view(c)


@app.delete("/api/cameras/{cid}")
def cam_delete(cid: str):
    _cam(cid)
    w = workers.pop(cid, None)
    if w:
        w.stop()
    cameras.pop(cid)
    save_cameras()
    return {"ok": True}


class CalibrationIn(BaseModel):
    img_pts: Optional[list] = None      # 4 điểm ảnh [[u,v],...]
    ground_pts: Optional[list] = None   # 4 điểm nền tương ứng (m)


@app.put("/api/cameras/{cid}/calibration")
def cam_calibrate(cid: str, body: CalibrationIn):
    c = _cam(cid)
    if body.img_pts:
        if len(body.img_pts) < 4 or len(body.img_pts) != len(body.ground_pts or []):
            raise HTTPException(400, "Cần ít nhất 4 cặp điểm")
        c["calibration"] = {"img_pts": body.img_pts, "ground_pts": body.ground_pts}
    else:
        c["calibration"] = None
    save_cameras()
    w = workers.get(cid)
    if w:
        w.apply_calibration()
    return cam_view(c)


@app.get("/api/cameras/{cid}/frame")
def cam_frame(cid: str):
    import cv2
    _cam(cid)
    w = workers.get(cid)
    if not w or w.raw is None:
        raise HTTPException(404, "Chưa có khung hình – hãy chạy camera trước")
    ok, jpg = cv2.imencode(".jpg", w.raw)
    return Response(jpg.tobytes(), media_type="image/jpeg")


@app.get("/api/cameras/{cid}/stream")
async def cam_stream(cid: str):
    _cam(cid)

    async def gen():
        last = None
        while True:
            w = workers.get(cid)
            if w is None:
                break
            jpg = w.jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            elif not w.is_alive() and jpg is last:
                # giữ khung cuối khi video đã chạy xong
                await asyncio.sleep(0.5)
                continue
            await asyncio.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ----------------------------------------------------------------- sự kiện
@app.get("/api/events")
def events(camera: str = "", type: str = "", severity: str = "", status: str = "",
           limit: int = 50, offset: int = 0):
    return store.list(camera, type, severity, status, min(limit, 500), offset)


@app.get("/api/events/{eid}")
def event(eid: int):
    e = store.get(eid)
    if not e:
        raise HTTPException(404)
    return e


class EventPatch(BaseModel):
    status: Optional[str] = None   # new | ack | false_alarm | resolved
    note: Optional[str] = None


@app.patch("/api/events/{eid}")
def event_patch(eid: int, body: EventPatch):
    if body.status and body.status not in {"new", "ack", "false_alarm", "resolved"}:
        raise HTTPException(400, "Trạng thái không hợp lệ")
    store.update(eid, body.status, body.note)
    e = store.get(eid)
    hub.broadcast({"kind": "event_update", "event": e})
    return e


@app.delete("/api/events")
def events_clear():
    store.clear()
    return {"ok": True}


@app.get("/media/snapshots/{name}")
def snapshot(name: str):
    p = (SNAP_DIR / Path(name).name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/media/clips/{name}")
def clip(name: str):
    p = (CLIP_DIR / Path(name).name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/webm" if p.suffix == ".webm" else "video/mp4")


@app.get("/api/stats")
def stats():
    return store.stats()


# ----------------------------------------------------------------- cài đặt
@app.get("/api/settings")
def get_settings():
    return SETTINGS.to_dict()


@app.put("/api/settings")
def put_settings(body: dict):
    errs = SETTINGS.update(body, strict=True)       # có lỗi thì không áp dụng gì cả
    if errs:
        raise HTTPException(400, "; ".join(errs))
    save_settings()
    return SETTINGS.to_dict()


# ----------------------------------------------------------------- websocket + UI
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(websocket)


app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
