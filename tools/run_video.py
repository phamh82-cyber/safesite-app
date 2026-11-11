"""Chạy một video không cần giao diện, xuất video đã vẽ + danh sách sự kiện.

    python -m tools.run_video input.mp4 --out output.mp4
"""
from __future__ import annotations

import argparse
import json

from app.config import load_settings
from app.pipeline import process_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--fps", type=float, help="Số khung hình phân tích mỗi giây")
    args = ap.parse_args()
    s = load_settings()
    s.pipeline.realtime_files = False
    if args.fps:
        s.pipeline.process_fps = args.fps
    evs = process_video(args.video, s, writer_path=args.out)
    for e in evs:
        print(f"{e['t']:7.2f}s  {e['type']:26} conf={e['confidence']:.2f}  {e['track_ids']}  "
              f"{json.dumps(e['details'], ensure_ascii=False)}")
    print(f"Tổng: {len(evs)} sự kiện → {args.out}")


if __name__ == "__main__":
    main()
