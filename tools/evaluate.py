"""Đánh giá theo SỰ KIỆN (không theo khung hình).

File nhãn CSV:   video,type,start,end
    video  – tên file trong thư mục --videos
    type   – FALL | NEAR_MISS | PERSON_VEHICLE_COLLISION | VEHICLE_COLLISION | ...
    start, end – giây trong video. Video không có sự kiện: ghi một dòng type=NONE.

Quy tắc khớp: một cảnh báo cùng loại nằm trong [start - 1 s, end + 3 s] là TP;
nhiều cảnh báo cho cùng một nhãn chỉ tính 1 TP, phần dư không tính FP.
FALL_IMMOBILE được gộp vào FALL; PERSON_DOWN tính riêng.

Ví dụ:
    python -m tools.evaluate --labels data/labels.csv --videos data/test_videos --out data/eval
    python -m tools.evaluate --labels labels.csv --videos vids --preds data/eval/predictions.csv  # chỉ chấm lại
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

ALIAS = {"FALL_IMMOBILE": "FALL"}
PRE, POST = 1.0, 3.0


def load_labels(path: Path):
    labels = defaultdict(list)
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            labels[r["video"]].append({"type": r["type"].strip().upper(),
                                       "start": float(r.get("start") or 0), "end": float(r.get("end") or 0)})
    return labels


def predict(videos: Path, names, out: Path):
    from app.config import load_settings
    from app.pipeline import process_video
    s = load_settings()
    s.pipeline.realtime_files = False
    (out / "settings_used.json").write_text(json.dumps(s.to_dict(), indent=2, ensure_ascii=False),
                                            encoding="utf-8")        # để thí nghiệm A–D tái lập được
    rows, durations = [], {}
    for name in names:
        t0 = time.time()
        evs = process_video(str(videos / name), s, writer_path=str(out / f"annotated_{Path(name).stem}.mp4"))
        durations[name] = time.time() - t0
        rows += [{"video": name, "type": e["type"], "t": round(e["t"], 2), "confidence": e["confidence"]}
                 for e in evs]
        print(f"{name}: {len(evs)} cảnh báo, {durations[name]:.1f}s xử lý")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "type", "t", "confidence"])
        w.writeheader()
        w.writerows(rows)
    return rows


def score(labels, preds, video_seconds: float | None = None):
    by_video = defaultdict(list)
    for p in preds:
        by_video[p["video"]].append({**p, "type": ALIAS.get(p["type"], p["type"]), "t": float(p["t"])})
    stat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "lat": []})
    for video, gts in labels.items():
        pr = by_video.get(video, [])
        used = set()
        for g in gts:
            if g["type"] == "NONE":
                continue
            hits = [i for i, p in enumerate(pr) if p["type"] == g["type"]
                    and g["start"] - PRE <= p["t"] <= g["end"] + POST]
            if hits:
                stat[g["type"]]["tp"] += 1
                stat[g["type"]]["lat"].append(min(pr[i]["t"] for i in hits) - g["start"])
                used.update(hits)
            else:
                stat[g["type"]]["fn"] += 1
        for i, p in enumerate(pr):
            if i in used:
                continue
            # cảnh báo dư nhưng vẫn nằm trong cửa sổ của một nhãn cùng loại -> bỏ qua
            if any(g["type"] == p["type"] and g["start"] - PRE <= p["t"] <= g["end"] + POST for g in gts):
                continue
            stat[p["type"]]["fp"] += 1
    report = {}
    for typ, s in sorted(stat.items()):
        P = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else None
        R = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else None
        F = 2 * P * R / (P + R) if P and R else None
        report[typ] = {"TP": s["tp"], "FP": s["fp"], "FN": s["fn"],
                       "precision": None if P is None else round(P, 3),
                       "recall": None if R is None else round(R, 3), "f1": None if F is None else round(F, 3),
                       "mean_latency_s": round(sum(s["lat"]) / len(s["lat"]), 2) if s["lat"] else None}
        if video_seconds:
            report[typ]["fp_per_hour"] = round(s["fp"] / (video_seconds / 3600), 2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--videos", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("data/eval"))
    ap.add_argument("--preds", type=Path, help="Dùng file dự đoán có sẵn thay vì chạy mô hình")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    if args.preds:
        with args.preds.open(encoding="utf-8") as f:
            preds = list(csv.DictReader(f))
    else:
        preds = predict(args.videos, list(labels), args.out)
    secs = None
    try:
        import cv2
        secs = 0.0
        for v in labels:
            cap = cv2.VideoCapture(str(args.videos / v))
            secs += (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 25)
            cap.release()
        secs = secs or None
    except Exception:
        pass
    rep = score(labels, preds, secs)
    (args.out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'Loại':28} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6} {'Trễ(s)':>7}")
    for typ, r in rep.items():
        f = lambda x: "  —  " if x is None else f"{x:.2f}"
        print(f"{typ:28} {r['TP']:>4} {r['FP']:>4} {r['FN']:>4} {f(r['precision']):>6} {f(r['recall']):>6} "
              f"{f(r['f1']):>6} {f(r['mean_latency_s']):>7}")


if __name__ == "__main__":
    main()
