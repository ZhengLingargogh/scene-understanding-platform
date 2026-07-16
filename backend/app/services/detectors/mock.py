"""Mock keypoint detector for UI / pipeline testing without GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.services.detectors.base import KeypointDetectionResult, KeypointDetector


class MockKeypointDetector(KeypointDetector):
    """Synthetic SuperPoint-shaped features (deterministic per image path)."""

    def __init__(self, model_id: str = "mock-extractor") -> None:
        self._model_id = model_id
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def detect(self, image_path: Path) -> KeypointDetectionResult:
        if not self._loaded:
            self.load()

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            proc_w = max(64, min(orig_w, 640))
            proc_h = max(64, min(orig_h, 480))

        rng = np.random.default_rng(hash(str(image_path.resolve())) % (2**32))
        n = int(rng.integers(80, 200))
        keypoints = np.column_stack(
            [
                rng.uniform(0, proc_w, size=n),
                rng.uniform(0, proc_h, size=n),
            ]
        ).astype(np.float32)
        scores = rng.uniform(0.3, 1.0, size=n).astype(np.float32)
        descriptors = rng.standard_normal((n, 256)).astype(np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8

        feats = {
            "keypoints": torch.from_numpy(keypoints)[None],
            "keypoint_scores": torch.from_numpy(scores)[None],
            "descriptors": torch.from_numpy(descriptors)[None].transpose(1, 2),
        }

        return KeypointDetectionResult(
            feats=feats,
            processed_shape=(proc_w, proc_h),
            original_size=(orig_w, orig_h),
        )
