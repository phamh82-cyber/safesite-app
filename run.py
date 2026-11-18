"""Khởi động dashboard:  python run.py   rồi mở http://localhost:8000"""
import argparse
import logging
import sys
import threading
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    log_dir = Path(__file__).resolve().parent / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "safesite.log", encoding="utf-8"),
        ],
    )

    def handle_thread_exception(args) -> None:
        logging.getLogger("safesite").error(
            "Thread %s bị dừng do exception chưa xử lý",
            args.thread.name if args.thread else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = handle_thread_exception
    sys.excepthook = lambda exc_type, exc_value, exc_tb: logging.getLogger("safesite").critical(
        "Tiến trình SafeSite dừng do exception chưa xử lý",
        exc_info=(exc_type, exc_value, exc_tb),
    )
    print(f"SafeSite đang chạy tại http://{a.host}:{a.port}")
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        logging.getLogger("safesite").warning(
            "Đang mở cho cả mạng LAN (%s) nhưng API CHƯA có xác thực: ai cùng mạng cũng xem được "
            "camera, xóa sự kiện, đổi cài đặt. Chỉ dùng trong mạng tin cậy.", a.host)
    uvicorn.run("app.main:app", host=a.host, port=a.port, log_level="warning")
