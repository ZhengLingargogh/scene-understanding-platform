"""SAM batch semantic segmentation via automatic mask generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from app.services.salad.image_utils import list_images
from app.services.salad.paths import resolve_path
from app.services.sam.engine import SAMEngine
from app.services.semantic_segmenters.base import ProgressCallback, SemanticSegmenter
from app.services.semantic_segmenters.store import save_segmentation_manifest


class SAMSemanticSegmenter(SemanticSegmenter):
    """Segment each dataset image with SAM automatic mask generator (largest region)."""

    def __init__(self, model_id: str = "sam") -> None:
        self._model_id = model_id
        self._engine = SAMEngine.get_shared()

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
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_images(input_dir)
        total = len(image_paths)
        records = []

        for index, image_path in enumerate(image_paths, start=1):
            if on_progress:
                on_progress(index - 1, total, f"SAM 分割 {image_path.name} ({index}/{total})")

            detail = self._engine.segment_largest_auto(image_path)
            mask = detail["mask"]
            mask_filename = f"{image_path.stem}_mask.png"
            mask_path = output_dir / mask_filename
            cv2.imwrite(str(mask_path), mask * 255)

            records.append(
                {
                    "filename": image_path.name,
                    "source_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "score": detail["score"],
                    "mask_count": detail.get("mask_count", 0),
                }
            )

            if on_progress:
                on_progress(index, total, f"已保存 {mask_filename} ({index}/{total})")

        save_segmentation_manifest(
            output_dir,
            model_id=self._model_id,
            dataset_id=dataset_id,
            image_records=records,
        )

        return {
            "status": "completed",
            "model_id": self._model_id,
            "dataset_id": dataset_id,
            "input_path": str(input_dir),
            "output_path": str(output_dir),
            "total": total,
            "message": f"SAM 语义分割完成（{total} 张）",
        }
