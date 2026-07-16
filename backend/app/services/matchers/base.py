"""Abstract base for image matching plugins (Inference)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.schemas.pipeline import ImageMatchingResult, RetrievalReference


class Matcher(ABC):
    """Match query image against retrieved reference images or a second query."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def infer(
        self,
        *,
        query_image_path: str,
        references: List[RetrievalReference],
        scene_id: Optional[str] = None,
    ) -> ImageMatchingResult:
        pass

    def infer_pair(
        self,
        *,
        image0_path: str,
        image1_path: str,
        scene_id: Optional[str] = None,
        detector_model_id: Optional[str] = None,
    ) -> ImageMatchingResult:
        """Match two uploaded images directly (default: treat image1 as sole reference)."""
        ref = RetrievalReference(
            id="pair-ref",
            label="Image 1",
            score=1.0,
            image_path=image1_path,
        )
        return self.infer(
            query_image_path=image0_path,
            references=[ref],
            scene_id=scene_id,
        )

    @property
    def is_loaded(self) -> bool:
        return False
