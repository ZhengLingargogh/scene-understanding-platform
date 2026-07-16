"""Abstract base for batch semantic segmentation plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

ProgressCallback = Callable[[int, int, str], None]


class SemanticSegmenter(ABC):
    """Run semantic segmentation over all images in a dataset directory."""

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
    def infer_dataset(
        self,
        *,
        dataset_id: str,
        input_path: str,
        output_path: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        pass

    @property
    def is_loaded(self) -> bool:
        return False
