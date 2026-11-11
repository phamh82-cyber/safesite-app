"""Cấu hình mặc định của hệ thống. Mọi ngưỡng đều có thể chỉnh trong trang Cài đặt
của dashboard (lưu vào data/settings.json)."""
from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger("safesite")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAP_DIR = DATA_DIR / "snapshots"
CLIP_DIR = DATA_DIR / "clips"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "events.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
CAMERAS_PATH = DATA_DIR / "cameras.json"

for _d in (DATA_DIR, SNAP_DIR, CLIP_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Lớp COCO được coi là phương tiện. COCO không có xe nâng / máy xúc:
# muốn nhận diện các lớp đó cần fine-tune (xem README).
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Chiều cao thật trung bình (m) dùng cho tự hiệu chỉnh tỉ lệ khi chưa có homography
CLASS_HEIGHT_M = {"person": 1.70, "bicycle": 1.10, "car": 1.50, "motorcycle": 1.20,
                  "bus": 3.20, "truck": 3.00}


@dataclass
class FallSettings:
    descent_speed: float = 0.8        # tốc độ rơi của hông, đơn vị H_ref / giây
    height_drop: float = 0.35         # tỉ lệ chiều cao giảm trong cửa sổ descent_window
    descent_window: float = 0.7       # giây
    lying_angle: float = 60.0         # góc thân (độ) coi là nằm
    upright_angle: float = 35.0       # góc thân coi là đứng
    lying_aspect: float = 1.3         # AR = w/h của bbox coi là nằm (dự phòng)
    confirm_window: float = 2.0       # từ lúc rơi tới lúc nằm tối đa (giây)
    lying_hold: float = 1.0           # nằm liên tục bao lâu thì xác nhận FALL
    immobile_spread: float = 0.04     # độ dao động vị trí/kích thước bbox (theo H_ref) dưới ngưỡng = bất động
    torso_rate: float = 25.0          # thân đổ ≥ 25° trong 0,5 s cũng tính là pha rơi
    immobile_after: float = 10.0      # bất động sau ngã bao lâu thì nâng mức
    person_down_after: float = 5.0    # nằm bất động không có pha rơi -> PERSON_DOWN
    recover_hold: float = 1.5         # đứng lại bao lâu thì kết thúc sự kiện


@dataclass
class CollisionSettings:
    near_distance: float = 2.0        # m
    near_ttc: float = 2.0             # s
    near_hold: float = 0.3            # điều kiện phải kéo dài (s)
    min_vehicle_speed: float = 0.5    # m/s, xe đứng yên không tính near-miss theo khoảng cách
    contact_distance: float = 0.4     # m, người - xe
    confirm_window: float = 2.0       # s, chờ dấu hiệu bất thường của người sau tiếp xúc
    push_accel: float = 4.0           # m/s^2, người bị đẩy
    vv_contact_distance: float = 0.3  # m, xe - xe
    vv_min_closing: float = 1.0       # m/s trước va chạm
    vv_speed_drop: float = 0.5        # tỉ lệ giảm tốc sau va chạm
    vv_heading_change: float = 30.0   # độ
    cooldown: float = 10.0            # s, không báo lặp cùng một cặp


@dataclass
class PipelineSettings:
    process_fps: float = 10.0         # số khung hình phân tích mỗi giây
    imgsz: int = 640
    person_conf: float = 0.35
    vehicle_conf: float = 0.35
    pose_model: str = "yolo11n-pose.pt"
    det_model: str = "yolo11n.pt"
    focal_ratio: float = 0.9          # f ≈ 0.9 × chiều rộng ảnh (tự hiệu chỉnh)
    clip_pre: float = 3.0             # giây video trước sự kiện
    clip_post: float = 3.0            # giây sau sự kiện
    show_bev: bool = True             # bản đồ nhìn từ trên xuống ở góc khung hình
    realtime_files: bool = True       # phát file video với tốc độ thật
    stabilize: bool = True            # bù rung camera (quan trọng với điện thoại cầm/kẹp)
    shake_level: float = 0.006        # rung > 0,6% chiều rộng ảnh/khung ⇒ coi là đang rung mạnh


@dataclass
class HeightFallSettings:
    """Ngã từ trên cao: rơi gần tự do ≥ min_drop chiều cao người (vượt xa ngã cùng mặt bằng ≈0,5)."""
    enabled: bool = True
    min_speed: float = 2.0            # chiều cao người / giây (rơi tự do 2 m ≈ 3,7 H/s)
    min_drop: float = 0.7             # độ rơi tối thiểu để vào pha rơi (đơn vị chiều cao người)
    min_accel: float = 1.5            # H/s², rơi tự do ≈ 5,8; đi cầu thang/leo thang ≈ 0
    max_growth: float = 1.3           # bbox to lên > 30% = đang chạy lại gần camera, không phải rơi
    window: float = 1.0               # giây
    outcome_window: float = 3.0       # giây chờ kết quả (nằm / mất dấu) sau khi vào pha rơi
    lost_confirm: float = 0.4         # mất dấu bao lâu trong pha rơi thì báo
    cooldown: float = 10.0            # giây


@dataclass
class ObjectSettings:
    """Vật rơi / bị vật rơi đập trúng, dựa trên trừ nền (MOG2) trên ảnh đã ổn định."""
    enabled: bool = True
    work_width: int = 320             # chiều rộng ảnh phân tích chuyển động
    warmup: float = 3.0               # giây đầu để mô hình nền học
    min_area: float = 0.0003          # tỉ lệ diện tích khung
    max_area: float = 0.03
    min_speed: float = 0.25           # chiều cao khung / giây
    min_drop: float = 0.12            # độ rơi tối thiểu (tỉ lệ chiều cao khung)
    max_lateral: float = 0.6          # |dx| ≤ max_lateral × dy: loại chim, vật bay ngang
    min_samples: int = 4
    gate: float = 0.25                # bán kính ghép vệt (tỉ lệ chiều cao khung)
    max_fg_ratio: float = 0.25        # nền đổi > 25% khung (đổi sáng, rung) ⇒ bỏ khung
    max_blobs: int = 25               # quá nhiều vệt (mưa, nhiễu) ⇒ bỏ khung
    contact_margin: float = 0.15      # mở rộng bbox người (tỉ lệ chiều cao người) khi xét va trúng
    confirm_window: float = 2.0       # giây chờ người phản ứng (ngã/mất dấu) sau khi bị trúng
    cooldown: float = 5.0


@dataclass
class HazardSettings:
    """Cháy / khói. Có mô hình YOLO fine-tune thì dùng mô hình; chưa có thì dùng bộ lọc màu + nhấp nháy."""
    enabled: bool = True
    fire_model: str = ""              # đường dẫn .pt (vd models/fire_smoke_best.pt); trống = không dùng
    conf: float = 0.4
    every: int = 3                    # chạy mô hình mỗi N khung đã phân tích
    window: float = 3.0               # giây
    min_hits: int = 3                 # số lần phát hiện cùng vị trí trong cửa sổ để báo
    cooldown: float = 30.0
    color_fallback: bool = True       # dùng bộ lọc màu khi chưa có mô hình (dễ báo nhầm hơn)
    color_min_area: float = 0.002
    color_max_area: float = 0.3
    color_flicker: float = 0.15       # vùng màu lửa phải thay đổi hình dạng (đèn, áo cam thì đứng yên)


# Miền giá trị hợp lệ (min, max). Tham số số không có ở đây chỉ cần hữu hạn và ≥ 0.
RANGES: dict = {
    "fall": {"descent_speed": (0.1, 10), "height_drop": (0.05, 1), "descent_window": (0.2, 3),
             "lying_angle": (30, 90), "upright_angle": (5, 60), "lying_aspect": (0.8, 3),
             "confirm_window": (0.2, 10), "lying_hold": (0.1, 10), "immobile_spread": (0.001, 0.5),
             "torso_rate": (5, 90), "immobile_after": (1, 600), "person_down_after": (1, 600),
             "recover_hold": (0.1, 60)},
    "collision": {"near_distance": (0.1, 20), "near_ttc": (0.1, 30), "near_hold": (0, 10),
                  "contact_distance": (0.05, 5), "vv_contact_distance": (0.05, 5),
                  "vv_speed_drop": (0.05, 1), "vv_heading_change": (5, 180), "cooldown": (0, 3600)},
    "pipeline": {"process_fps": (1, 60), "imgsz": (160, 2048), "person_conf": (0.05, 0.95),
                 "vehicle_conf": (0.05, 0.95), "focal_ratio": (0.3, 3), "clip_pre": (0, 60),
                 "clip_post": (0, 60), "shake_level": (0.0005, 0.2)},
    "height_fall": {"min_speed": (0.5, 20), "min_drop": (0.3, 5), "min_accel": (0, 20),
                    "max_growth": (1.0, 3.0), "window": (0.3, 3), "outcome_window": (0.5, 30)},
    "objects": {"work_width": (64, 1280), "min_area": (0.00001, 0.1), "max_area": (0.0005, 0.5),
                "min_speed": (0.02, 10), "min_drop": (0.01, 1), "max_lateral": (0, 5),
                "min_samples": (2, 30), "gate": (0.02, 1), "max_fg_ratio": (0.02, 1),
                "max_blobs": (1, 500), "contact_margin": (0, 2)},
    "hazard": {"conf": (0.05, 0.95), "every": (1, 60), "window": (0.5, 60), "min_hits": (1, 60),
               "color_min_area": (0.0001, 0.5), "color_max_area": (0.001, 1), "color_flicker": (0, 1)},
}
MODEL_FIELDS = {"pose_model", "det_model", "fire_model"}


def _coerce(section: str, key: str, cur, v):
    """Ép kiểu và kiểm tra miền giá trị. Ném ValueError với thông báo tiếng Việt."""
    if isinstance(cur, bool):
        if not isinstance(v, bool):
            raise ValueError("phải là bật/tắt")
        return v
    if isinstance(cur, str):
        if not isinstance(v, str) or any(c in v for c in "\r\n\x00"):
            raise ValueError("phải là chuỗi một dòng")
        v = v.strip()
        if key in MODEL_FIELDS and v and not v.lower().endswith(".pt"):
            raise ValueError("phải là file .pt")
        if key in {"pose_model", "det_model"} and not v:
            raise ValueError("không được để trống")
        return v
    if isinstance(v, bool) or v is None:
        raise ValueError("phải là số")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError("phải là số")
    if not math.isfinite(x):
        raise ValueError("phải là số hữu hạn")
    if isinstance(cur, int):
        if x != int(x):
            raise ValueError("phải là số nguyên")
        x = int(x)
    lo, hi = RANGES.get(section, {}).get(key, (0, None))
    if x < lo or (hi is not None and x > hi):
        rng = f"{lo}–{hi}" if hi is not None else f"≥ {lo}"
        raise ValueError(f"ngoài khoảng cho phép ({rng})")
    return x


@dataclass
class Settings:
    fall: FallSettings = field(default_factory=FallSettings)
    collision: CollisionSettings = field(default_factory=CollisionSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    height_fall: HeightFallSettings = field(default_factory=HeightFallSettings)
    objects: ObjectSettings = field(default_factory=ObjectSettings)
    hazard: HazardSettings = field(default_factory=HazardSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, data: dict, strict: bool = False) -> list:
        """Cập nhật từ dict. Trả về danh sách lỗi. strict=True: có lỗi thì KHÔNG áp dụng gì cả;
        strict=False (đọc file cũ): bỏ qua giá trị sai, áp dụng phần còn lại."""
        errors, staged = [], []
        for f in fields(self):
            sec = data.get(f.name)
            if not isinstance(sec, dict):
                continue
            obj = getattr(self, f.name)
            for k, v in sec.items():
                if not hasattr(obj, k):
                    continue
                try:
                    staged.append((obj, k, _coerce(f.name, k, getattr(obj, k), v)))
                except ValueError as e:
                    errors.append(f"{f.name}.{k}: {e}")
        if errors and strict:
            return errors
        for obj, k, v in staged:
            setattr(obj, k, v)
        return errors


_lock = threading.Lock()
SETTINGS = Settings()


def load_settings() -> Settings:
    if SETTINGS_PATH.exists():
        try:
            for e in SETTINGS.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))):
                log.warning("settings.json: bỏ qua giá trị sai – %s", e)
        except Exception:  # file hỏng -> dùng mặc định
            log.warning("settings.json hỏng, dùng cài đặt mặc định")
    return SETTINGS


def save_settings() -> None:
    with _lock:
        SETTINGS_PATH.write_text(json.dumps(SETTINGS.to_dict(), indent=2, ensure_ascii=False),
                                 encoding="utf-8")
