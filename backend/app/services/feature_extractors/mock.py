"""Mock feature extractor for Benchmark (simulated progress, no real CV)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.services.feature_extractors.base import FeatureExtractor, ProgressCallback

MOCK_IMAGE_COUNT = 100
MOCK_STEP_MS = 80


class MockFeatureExtractor(FeatureExtractor):
    """Placeholder extractor — simulates batch feature extraction with progress."""

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

    def infer_dataset(
        self,
        *,
        dataset_id: str,
        input_path: str,
        output_path: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        if not self._loaded:
            self.load()

        total = MOCK_IMAGE_COUNT
        for i in range(1, total + 1):
            time.sleep(MOCK_STEP_MS / 1000.0)
            message = "Extracting features… {}/{}".format(i, total)
            if on_progress:
                on_progress(i, total, message)

        return {
            "status": "completed",
            "model_id": self._model_id,
            "dataset_id": dataset_id,
            "input_path": input_path,
            "output_path": output_path,
            "total": total,
            "message": "Feature extraction completed",
        }
