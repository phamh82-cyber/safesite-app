"""YOLO11-Pose (người + keypoints) và YOLO11 object (xe cơ giới, lửa).

Lưu ý: yolo11n.pt gốc (COCO) chỉ có car/truck/bus/motorcycle, không có forklift/fire.
Nếu bạn dùng weight tự train, chỉ cần tên class nằm trong Settings.vehicle_classes / fire_classes.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Det:
    box: tuple            # x1,y1,x2,y2
    conf: float
    name: str = ""
    kpts: Optional[np.ndarray] = None   # (17,3) x,y,conf


def calc_torso_angle(kpts, box=None, kconf: float = 0.3) -> Optional[float]:
    """Góc nghiêng thân so với phương thẳng đứng (độ): 0 = đứng thẳng, 90 = nằm ngang.

    Dùng trung điểm vai (5,6) và hông (11,12). Thiếu keypoint -> suy từ tỉ lệ bbox.
    """
    if kpts is not None:
        sh = [kpts[i] for i in (5, 6) if kpts[i][2] >= kconf]
        hp = [kpts[i] for i in (11, 12) if kpts[i][2] >= kconf]
        if sh and hp:
            sx, sy = np.mean([p[0] for p in sh]), np.mean([p[1] for p in sh])
            hx, hy = np.mean([p[0] for p in hp]), np.mean([p[1] for p in hp])
            return math.degrees(math.atan2(abs(sx - hx), abs(sy - hy) + 1e-6))
    if box is not None:
        w, h = box[2] - box[0], box[3] - box[1]
        if w > 1.2 * h:
            return 70.0
    return None


class YoloDetector:
    def __init__(self, pose_weights, obj_weights, device: Optional[str] = None):
        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")
        self.lock = threading.Lock()
        self.pose = YOLO(str(pose_weights)) if pose_weights else None
        self.obj = YOLO(str(obj_weights)) if obj_weights else None

    def detect(self, frame, s):
        """Trả về (persons, vehicles, fires) — mỗi phần tử là Det."""
        persons, vehicles, fires = [], [], []
        kw = dict(imgsz=s.yolo_imgsz, conf=s.yolo_conf, device=self.device,
                  half=self.half, verbose=False)
        veh = {c.lower() for c in s.vehicle_classes}
        fire = {c.lower() for c in s.fire_classes}
        with self.lock:
            if self.pose is not None:
                r = self.pose(frame, classes=[0], **kw)[0]
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    conf = r.boxes.conf.cpu().numpy()
                    kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else [None] * len(conf)
                    for b, c, k in zip(xyxy, conf, kp):
                        persons.append(Det(tuple(map(float, b)), float(c), "person", k))
            if self.obj is not None:
                r = self.obj(frame, **kw)[0]
                if r.boxes is not None and len(r.boxes):
                    names = r.names
                    for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy()):
                        n = str(names[int(k)]).lower()
                        d = Det(tuple(map(float, b)), float(c), n)
                        if n in veh:
                            vehicles.append(d)
                        elif n in fire:
                            fires.append(d)
        return persons, vehicles, fires
