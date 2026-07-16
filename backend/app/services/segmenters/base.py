"""Abstract base for interactive segmentation plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    """Point-prompt segmentation output."""

    mask: np.ndarray
    contour: List[List[float]]
    score: float
    backend: str
    message: str = ""


class Segmenter(ABC):
    """Segment an image from a foreground point prompt (x, y)."""

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
    def segment_point(
        self,
        *,
        image_path: Path,
        point_x: float,
        point_y: float,
        image_width: int,
        image_height: int,
    ) -> SegmentationResult:
        pass

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def is_implemented(self) -> bool:
        """Whether real inference is wired (vs registry placeholder)."""
        return True
