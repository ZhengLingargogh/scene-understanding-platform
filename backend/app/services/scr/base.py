"""Abstract base for Scene Coordinate Regression plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SCRModel(ABC):
    """Predict scene coordinates / camera pose from a single query image."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Registered model identifier."""

    @abstractmethod
    def load(self) -> None:
        """Load network weights and scene heads."""

    @abstractmethod
    def unload(self) -> None:
        """Release GPU / CPU resources."""

    @abstractmethod
    def infer(
        self,
        *,
        image_path: str,
        scene_id: str,
        focal_length: Optional[float] = None,
        pp_x: Optional[float] = None,
        pp_y: Optional[float] = None,
        gt_pose: Optional[list] = None,
        calibration: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Run coordinate regression + PnP; return pose dict."""

    @classmethod
    def list_scenes(cls) -> List[Dict[str, str]]:
        """Optional scene → weights mapping for this SCR backend."""
        return []

    @property
    def is_loaded(self) -> bool:
        return False
