"""Cấu hình hệ thống SafeSite.

- Đường dẫn model / storage: hằng số.
- Các thông số vận hành (ngưỡng, góc ngã, bật/tắt module, danh sách camera):
  lưu trong settings.json và chỉnh được từ giao diện Settings, áp dụng nóng (không cần restart).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STORAGE_DIR = BASE_DIR / "storage"
CLIPS_DIR = STORAGE_DIR / "clips"
SNAPSHOTS_DIR = STORAGE_DIR / "snapshots"
UPLOADS_DIR = STORAGE_DIR / "uploads"
DB_PATH = STORAGE_DIR / "events.db"
SETTINGS_PATH = BASE_DIR / "settings.json"

VIDEOMAE_WEIGHTS = MODELS_DIR / "best_multitask_videomae.pth"
ACTION_VOCAB = MODELS_DIR / "action_vocab.json"
YOLO_POSE_WEIGHTS = MODELS_DIR / "yolo11n-pose.pt"
YOLO_OBJ_WEIGHTS = MODELS_DIR / "yolo11n.pt"

CLIP_LEN = 16      # số frame đưa vào VideoMAE
BUFFER_LEN = 90    # ~3 giây tại 30 FPS


@dataclass
class CameraConfig:
    id: str
    name: str
    source: str            # rtsp://..., đường dẫn file, hoặc số (webcam: "0")
    zone: str = ""
    enabled: bool = True


@dataclass
class Settings:
    # --- Module ---
    enable_yolo: bool = True
    enable_videomae: bool = True
    # --- VideoMAE / Fusion ---
    videomae_threshold: float = 0.50      # Hazard threshold
    consensus_ratio: float = 0.70         # p >= thr*ratio và YOLO đồng thuận => vẫn xác thực
    verify_interval_s: float = 1.0        # tối thiểu giữa 2 lần chạy VideoMAE trên 1 camera
    cooldown_s: float = 10.0              # chống spam theo Track ID
    roi_margin: float = 0.30
    # --- YOLO / Candidate ---
    yolo_conf: float = 0.35
    yolo_imgsz: int = 640
    lying_angle_deg: float = 55.0         # LYING_ANGLE_DEG
    fall_min_frames: int = 3              # số frame liên tiếp nghiêng > góc ngã
    safe_distance: float = 1.0            # khoảng cách mép người-xe, đơn vị = chiều cao người
    vehicle_classes: list = field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle", "forklift", "excavator", "bulldozer",
        "crane", "loader", "dump truck", "vehicle"])
    fire_classes: list = field(default_factory=lambda: ["fire", "flame", "smoke"])
    # --- Evidence ---
    clip_pre_s: float = 2.5
    clip_post_s: float = 2.5
    # --- Plug-in class VideoMAE của bạn: "package.module:callable" (xem videomae.py) ---
    videomae_factory: str = ""


class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.settings = Settings()
        self.cameras: list[CameraConfig] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.update_settings(data.get("settings", {}), save=False)
        self.cameras = [CameraConfig(**c) for c in data.get("cameras", [])]

    def save(self) -> None:
        with self.lock:
            payload = {"settings": asdict(self.settings),
                       "cameras": [asdict(c) for c in self.cameras]}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    def update_settings(self, patch: dict, save: bool = True) -> None:
        with self.lock:
            for f in fields(Settings):
                if f.name not in patch or patch[f.name] is None:
                    continue
                cur, new = getattr(self.settings, f.name), patch[f.name]
                try:
                    if isinstance(cur, bool):
                        new = bool(new)
                    elif isinstance(cur, int):
                        new = int(new)
                    elif isinstance(cur, float):
                        new = float(new)
                    elif isinstance(cur, str):
                        new = str(new)
                    elif isinstance(cur, list):
                        new = [str(x).strip() for x in new if str(x).strip()]
                except (TypeError, ValueError):
                    continue
                setattr(self.settings, f.name, new)
            if save:
                self.save()

    def set_cameras(self, cams: list[CameraConfig]) -> None:
        with self.lock:
            self.cameras = list(cams)
            self.save()
