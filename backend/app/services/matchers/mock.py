"""Mock matcher for Inference."""

from __future__ import annotations

from typing import List, Optional

from app.schemas.pipeline import ImageMatchingResult, RetrievalReference
from app.services.matchers.base import Matcher


class MockMatcher(Matcher):
    def __init__(self, model_id: str = "mock-matcher") -> None:
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

    def infer(
        self,
        *,
        query_image_path: str,
        references: List[RetrievalReference],
        scene_id: Optional[str] = None,
    ) -> ImageMatchingResult:
        if not self._loaded:
            self.load()
        ref_count = len(references) or 8
        return ImageMatchingResult(
            status="completed",
            message="Mock geometric matching on {} references".format(ref_count),
            match_count=ref_count * 12,
            inlier_count=0,
            inlier_ratio=0.0,
        )
