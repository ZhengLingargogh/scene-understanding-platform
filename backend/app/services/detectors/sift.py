"""OpenCV SIFT keypoint detector plugin."""

from __future__ import annotations

from pathlib import Path

import torch

from app.services.detectors.base import KeypointDetectionResult, KeypointDetector
from app.services.sift.engine import SIFTEngine


class SIFTDetector(KeypointDetector):
    def __init__(self, model_id: str = "sift") -> None:
        self._model_id = model_id
        self._engine = SIFTEngine.get_shared()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    def load(self) -> None:
        self._engine.load()

    def unload(self) -> None:
        self._engine.unload()

    def detect(self, image_path: Path) -> KeypointDetectionResult:
        self.load()
        data = self._engine.extract_image(image_path)
        width, height = int(data["image_size"][0]), int(data["image_size"][1])
        keypoints = data["keypoints"]
        scores = data["scores"]
        descriptors = data["descriptors"]

        feats = {
            "keypoints": torch.from_numpy(keypoints.astype("float32"))[None],
            "keypoint_scores": torch.from_numpy(scores.astype("float32"))[None],
            "descriptors": torch.from_numpy(descriptors.astype("float32"))[None],
        }

        return KeypointDetectionResult(
            feats=feats,
            processed_shape=(width, height),
            original_size=(width, height),
            backend="sift",
        )
