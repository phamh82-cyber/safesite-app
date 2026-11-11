"""Khởi động dashboard:  python run.py   rồi mở http://localhost:8000"""
import argparse
import logging

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"SafeSite đang chạy tại http://{a.host}:{a.port}")
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        logging.getLogger("safesite").warning(
            "Đang mở cho cả mạng LAN (%s) nhưng API CHƯA có xác thực: ai cùng mạng cũng xem được "
            "camera, xóa sự kiện, đổi cài đặt. Chỉ dùng trong mạng tin cậy.", a.host)
    uvicorn.run("app.main:app", host=a.host, port=a.port, log_level="warning")
