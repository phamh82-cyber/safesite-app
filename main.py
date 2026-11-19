"""SafeSite – điểm khởi chạy.   python main.py [--host 0.0.0.0] [--port 8000] [--source video.mp4]"""
import argparse
import os

# RTSP qua TCP ổn định hơn UDP; phải đặt trước khi OpenCV mở stream đầu tiên.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import uvicorn  # noqa: E402

from app.config import CameraConfig, SettingsStore  # noqa: E402
from app.web.server import create_app  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--source", help="thêm nhanh 1 camera: rtsp://..., file video, hoặc 0 (webcam)")
    args = ap.parse_args()

    store = SettingsStore()
    if args.source:
        if not any(c.source == args.source for c in store.cameras):
            store.set_cameras([*store.cameras, CameraConfig(id=f"cam{len(store.cameras) + 1}",
                                                            name=f"Camera {len(store.cameras) + 1}",
                                                            source=args.source)])
    uvicorn.run(create_app(store), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
