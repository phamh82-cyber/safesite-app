"""Bộ đệm thời gian thực.

ClipBuffer     : ring buffer 32 frame gần nhất (full-res) -> lấy 16 frame cho VideoMAE (full + ROI).
EvidenceBuffer : ring buffer JPEG vài giây gần nhất, dùng để cắt clip minh chứng (trước sự cố).
"""
from __future__ import annotations

import threading
from collections import deque

import cv2
import numpy as np

from app.config import BUFFER_LEN, CLIP_LEN


def union_roi(boxes, frame_shape, margin: float = 0.30, square: bool = True):
    """Union box của các đối tượng liên quan, mở rộng lề `margin`, (tuỳ chọn) ép gần vuông."""
    if not boxes:
        return None
    h, w = frame_shape[:2]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    if square:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1)
        x1, x2, y1, y2 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


class ClipBuffer:
    def __init__(self, size: int = BUFFER_LEN, clip_len: int = CLIP_LEN):
        self.size = size
        self.clip_len = clip_len
        self._buf: deque = deque(maxlen=size)
        self._lock = threading.Lock()

    def push(self, ts: float, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append((ts, frame))

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, roi=None):
        """Trả về (full_frames, roi_frames|None): mỗi list gồm `clip_len` frame BGR, đều theo thời gian."""
        with self._lock:
            items = list(self._buf)
        if not items:
            return [], None
        idx = np.linspace(0, len(items) - 1, self.clip_len).round().astype(int)
        full = [items[i][1] for i in idx]
        crops = None
        if roi is not None:
            x1, y1, x2, y2 = roi
            crops = [f[y1:y2, x1:x2].copy() for f in full]
        return full, crops


class EvidenceBuffer:
    def __init__(self, max_seconds: float = 6.0):
        self.max_seconds = max_seconds
        self._buf: deque = deque()
        self._lock = threading.Lock()

    def push(self, ts: float, jpeg: bytes) -> None:
        with self._lock:
            self._buf.append((ts, jpeg))
            while self._buf and ts - self._buf[0][0] > self.max_seconds:
                self._buf.popleft()

    def last(self, seconds: float, now: float):
        with self._lock:
            return [(t, j) for t, j in self._buf if now - t <= seconds]


def encode_jpeg(frame: np.ndarray, max_width: int = 960, quality: int = 75) -> bytes:
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
