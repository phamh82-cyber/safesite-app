"""Wrapper MultiTaskVideoMAE: nạp best_multitask_videomae.pth + action_vocab.json, suy luận FP16.

Đầu ra mô hình (2 task): hazard (nhị phân 2 lớp) + action (đa nhãn multi-label), dùng Attention Pooling.
"""
from __future__ import annotations

import importlib
import json
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_vocab(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "action2idx" in data:
        data = data["action2idx"]
    action2idx = {str(k): int(v) for k, v in data.items()}
    idx2action = {v: k for k, v in action2idx.items()}
    return action2idx, idx2action


def build_default_model(num_actions: int):
    import torch
    import torch.nn as nn
    from transformers import VideoMAEConfig, VideoMAEModel

    class MultiTaskVideoMAE(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = VideoMAEConfig(num_frames=16, image_size=224)
            self.videomae = VideoMAEModel(cfg)
            hidden_size = cfg.hidden_size

            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

            self.hazard_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
            )

            self.action_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, num_actions),
            )

        def forward(self, pixel_values):
            outputs = self.videomae(pixel_values=pixel_values)
            sequence_output = outputs.last_hidden_state

            attn_weights = torch.softmax(self.attention(sequence_output), dim=1)
            pooled_output = torch.sum(sequence_output * attn_weights, dim=1)

            return self.hazard_head(pooled_output), self.action_head(pooled_output)

    return MultiTaskVideoMAE()


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


class MultiTaskVideoMAEPipeline:
    def __init__(self, weights_path, vocab_path, factory: str = "", device: Optional[str] = None,
                 image_size: int = 224):
        import torch

        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fp16 = self.device.startswith("cuda")
        self.image_size = image_size
        self.lock = threading.Lock()
        self.action2idx, self.idx2action = load_vocab(vocab_path)
        n = len(self.action2idx)

        if factory:
            mod, _, fn = factory.partition(":")
            builder = getattr(importlib.import_module(mod), fn)
            self.model = builder(weights_path=str(weights_path), num_actions=n,
                                 idx2action=self.idx2action, device=self.device)
        else:
            self.model = build_default_model(n)
            self._load_weights(weights_path)

        self.model.to(self.device).eval()
        if self.fp16:
            self.model.half()
        self.warmup()

    def _load_weights(self, path):
        torch = self.torch
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(ckpt)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

        # Chuẩn hóa tên key attention bias cho phiên bản transformers mới
        sd = {
            k.replace(".attention.q_bias", ".attention.query.bias").replace(
                ".attention.v_bias", ".attention.value.bias"
            ): v
            for k, v in sd.items()
        }

        res = self.model.load_state_dict(sd, strict=False)
        total = len(self.model.state_dict())
        miss = len(res.missing_keys)
        print(f"[videomae] nạp weight: thiếu {miss}/{total} key, thừa {len(res.unexpected_keys)} key")
        if miss > 0.2 * total or (res.unexpected_keys and len(res.unexpected_keys) > 0.2 * total):
            raise RuntimeError(
                "Kiến trúc không khớp file .pth "
                f"(missing ví dụ: {res.missing_keys[:3]}, unexpected ví dụ: {res.unexpected_keys[:3]}).")

    def warmup(self):
        dummy = [np.zeros((self.image_size, self.image_size, 3), np.uint8)] * 16
        self.predict([dummy])

    def _prep(self, frames):
        arr = np.stack([cv2.cvtColor(cv2.resize(f, (self.image_size, self.image_size),
                                                interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                        for f in frames]).astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return arr.transpose(0, 3, 1, 2)   # T,C,H,W

    def _parse(self, out):
        torch = self.torch
        if isinstance(out, dict):
            hz = next(v for k, v in out.items() if "hazard" in k.lower())
            ac = next(v for k, v in out.items() if "action" in k.lower())
        else:
            hz, ac = out[0], out[1]
        hz = hz.float()
        if hz.ndim == 2 and hz.shape[1] == 2:
            p = torch.softmax(hz, dim=1)[:, 1]
        else:
            p = torch.sigmoid(hz.reshape(hz.shape[0], -1)[:, 0])
        # Dùng sigmoid cho action_head vì lúc train dùng BCEWithLogitsLoss
        return p, torch.sigmoid(ac.float())

    def predict(self, views: list):
        """views: list các clip (mỗi clip = list 16 frame BGR), vd [full, roi]. Lấy view có hazard cao nhất."""
        torch = self.torch
        x = torch.from_numpy(np.stack([self._prep(v) for v in views])).to(self.device)
        if self.fp16:
            x = x.half()
        with self.lock, torch.inference_mode():
            p, a = self._parse(self.model(x))
        p, a = p.cpu().numpy(), a.cpu().numpy()
        best = int(p.argmax())
        k = int(a[best].argmax())
        return {
            "hazard_prob": float(p[best]),
            "top_action": self.idx2action.get(k, str(k)),
            "action_conf": float(a[best][k]),
            "view": "full" if best == 0 else "roi",
            "per_view": [float(v) for v in p],
        }