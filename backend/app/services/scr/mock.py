"""Mock SCR for tests / environments without ACE weights."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.scr.base import SCRModel


class MockSCRModel(SCRModel):
    """Returns a fixed pose without running deep learning."""

    def __init__(self, model_id: str = "mock-scr") -> None:
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
        image_path: str,
        scene_id: str,
        focal_length: Optional[float] = None,
        pp_x: Optional[float] = None,
        pp_y: Optional[float] = None,
        gt_pose: Optional[list] = None,
        calibration: Optional[dict] = None,
    ) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        return {
            "status": "completed",
            "pose_matrix": identity,
            "rotation_matrix": [row[:3] for row in identity[:3]],
            "translation": [0.0, 0.0, 0.0],
            "rotation_vector": [0.0, 0.0, 0.0],
            "inlier_count": 42,
            "confidence": 42.0,
            "pose_backend": "mock",
            "scene_id": scene_id,
            "image_path": image_path,
        }

    @classmethod
    def list_scenes(cls) -> List[Dict[str, str]]:
        return [{"scene_id": "nature", "weights_file": "mock_nature.pt"}]
