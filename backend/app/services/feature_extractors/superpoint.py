"""SuperPoint local feature extraction plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from app.services.feature_extractors.base import FeatureExtractor, ProgressCallback
from app.services.salad.image_utils import build_image_records, list_images
from app.services.salad.paths import resolve_path
from app.services.superpoint.engine import SuperPointEngine
from app.services.superpoint.feature_store import (
    features_dir_for,
    save_feature_index,
    save_image_features,
)


class SuperPointFeatureExtractor(FeatureExtractor):
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
        feat_dir = features_dir_for(output_dir)

        def _progress(done: int, total: int, msg: str) -> None:
            if on_progress:
                on_progress(done, total, msg)

        for index, (record, img_path) in enumerate(zip(records, image_paths), start=1):
            data = self._engine.extract_image(img_path)
            save_image_features(feat_dir, record["id"], data)
            record["feature_path"] = str(feat_dir / "{}.npz".format(record["id"]))
            _progress(index, len(image_paths), "Extracting SuperPoint features… {}/{}".format(index, len(image_paths)))

        save_feature_index(output_dir, model_id=self._model_id, image_records=records)

        return {
            "status": "completed",
            "model_id": self._model_id,
            "dataset_id": dataset_id,
            "input_path": str(input_dir),
            "output_path": str(output_dir),
            "total": len(image_paths),
            "feature_type": "local_superpoint",
            "message": "SuperPoint feature extraction completed",
        }
