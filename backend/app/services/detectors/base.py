"""Abstract base for keypoint detection plugins (image matching)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class KeypointDetectionResult:
    """Detector output consumed by matchers (e.g. LightGlue, OpenCV BF)."""

    feats: Dict[str, torch.Tensor]
    processed_shape: Tuple[int, int]
    original_size: Tuple[int, int]
    backend: str = "superpoint"


class KeypointDetector(ABC):
    """Detect sparse keypoints + descriptors on a single image."""

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
    def detect(self, image_path: Path) -> KeypointDetectionResult:
        pass

    @property
    def is_loaded(self) -> bool:
        return False
