"""SALAD global descriptor extraction plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from app.services.feature_extractors.base import FeatureExtractor, ProgressCallback
from app.services.salad.engine import SaladEngine
from app.services.salad.feature_store import save_feature_index
from app.services.salad.image_utils import build_image_records, list_images
from app.services.salad.paths import resolve_path


class SaladFeatureExtractor(FeatureExtractor):
    """Extract SALAD descriptors for all reference images in a directory."""

    def __init__(self, model_id: str = "salad") -> None:
        self._model_id = model_id
        self._engine = SaladEngine.get_shared()

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

    def infer_dataset(
        self,
        *,
        dataset_id: str,
        input_path: str,
        output_path: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        if not self._engine.is_loaded:
            self.load()

        input_dir = resolve_path(input_path)
        output_dir = resolve_path(output_path)
        image_paths = list_images(input_dir)
        records = build_image_records(image_paths)

        descriptors = self._engine.extract_paths(
            image_paths,
            on_progress=on_progress,
        )
        save_feature_index(
            output_dir,
            model_id=self._model_id,
            image_records=records,
            descriptors=descriptors,
            descriptor_dim=self._engine.descriptor_dim,
        )

        return {
            "status": "completed",
            "model_id": self._model_id,
            "dataset_id": dataset_id,
            "input_path": str(input_dir),
            "output_path": str(output_dir),
            "total": len(image_paths),
            "descriptor_dim": self._engine.descriptor_dim,
            "message": "SALAD feature extraction completed",
        }
