"""SuperPoint keypoint detector plugin."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.services.detectors.base import KeypointDetectionResult, KeypointDetector
from app.services.superpoint.engine import SuperPointEngine


class SuperPointDetector(KeypointDetector):
    def __init__(self, model_id: str = "superpoint") -> None:
        self._model_id = model_id
        self._engine = SuperPointEngine.get_shared()

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
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        tensor = self._engine.load_tensor_from_path(image_path)
        proc_w, proc_h = int(tensor.shape[2]), int(tensor.shape[1])
        feats = self._engine.extract_tensor(tensor)

        return KeypointDetectionResult(
            feats=feats,
            processed_shape=(proc_w, proc_h),
            original_size=(orig_w, orig_h),
        )
