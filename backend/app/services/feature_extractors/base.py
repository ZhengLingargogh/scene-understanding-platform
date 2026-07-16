"""Abstract base for feature extraction plugins (Benchmark)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


ProgressCallback = Callable[[int, int, str], None]


class FeatureExtractor(ABC):
    """Extract image descriptors / embeddings for a dataset."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Registered model identifier."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights and resources into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @abstractmethod
    def infer_dataset(
        self,
        *,
        dataset_id: str,
        input_path: str,
        output_path: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Run feature extraction over a dataset directory.

        Returns a summary dict when finished (status, total, output_path, …).
        """

    @property
    def is_loaded(self) -> bool:
        return False
