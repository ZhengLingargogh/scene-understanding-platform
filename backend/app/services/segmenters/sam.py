"""Original SAM interactive segmenter plugin."""

from __future__ import annotations

from pathlib import Path

from app.services.sam.engine import SAMEngine
from app.services.segmenters.base import Segmenter, SegmentationResult


class SAMSegmenter(Segmenter):
    def __init__(self, model_id: str = "sam") -> None:
        self._model_id = model_id
        self._engine = SAMEngine.get_shared()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    @property
    def is_implemented(self) -> bool:
        return True

    def load(self) -> None:
        self._engine.load()

    def unload(self) -> None:
        self._engine.unload()

    def prepare_image(self, image_path: Path) -> None:
        """Precompute image embedding for interactive point prompts."""
        self._engine.prepare_image(image_path)

    def segment_point(
        self,
        *,
        image_path: Path,
        point_x: float,
        point_y: float,
        image_width: int,
        image_height: int,
    ) -> SegmentationResult:
        detail = self._engine.segment_point(image_path, point_x, point_y)
        mask = detail["mask"]
        if mask.shape[0] != image_height or mask.shape[1] != image_width:
            import cv2

            mask = cv2.resize(
                mask.astype("uint8"),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask > 0).astype("uint8")
            from app.services.sam.engine import _largest_contour

            contour = _largest_contour(mask)
        else:
            contour = detail["contour"]

        return SegmentationResult(
            mask=mask,
            contour=contour,  # type: ignore[arg-type]
            score=float(detail["score"]),
            backend=self._model_id,
            message=str(detail["message"]),
        )
