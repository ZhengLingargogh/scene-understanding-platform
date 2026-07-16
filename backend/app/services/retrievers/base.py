"""Abstract base for image retrieval plugins (Inference)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.schemas.pipeline import ImageRetrievalResult


class Retriever(ABC):
    """Retrieve top-K reference images from a scene dataset gallery."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Registered model identifier."""

    @abstractmethod
    def load(self) -> None:
        """Load encoder / Faiss index / gallery metadata."""

    @abstractmethod
    def unload(self) -> None:
        """Release retrieval resources."""

    @abstractmethod
    def infer(
        self,
        *,
        image_path: str,
        dataset_id: str,
        top_k: int = 8,
        scene_id: Optional[str] = None,
    ) -> ImageRetrievalResult:
        """Retrieve similar images for a single query image."""

    @property
    def is_loaded(self) -> bool:
        return False
