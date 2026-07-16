"""ACE implementation of SCRModel (wraps existing LocalizationService)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services.localization import LocalizationService
from app.services.scene_constants import DEFAULT_SCENE_ID
from app.services.scr.base import SCRModel


class AceSCRModel(SCRModel):
    """
    ACE scene coordinate regression.

    Future SCR plugins: GLACESCRModel, CoCoSCRModel, …
    """

    def __init__(self, model_id: str = "ace") -> None:
        self._model_id = model_id
        self._engine: Optional[LocalizationService] = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> None:
        if self._engine is not None:
            return
        self._engine = LocalizationService(
            models_dir=settings.ace_models_dir,
            encoder_path=settings.ace_encoder_path,
            device=settings.ace_device,
            image_height=settings.ace_image_height,
        )
        self._engine.preload_scene(DEFAULT_SCENE_ID)

    def unload(self) -> None:
        self._engine = None

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
        if self._engine is None:
            self.load()
        assert self._engine is not None
        return self._engine.localize(
            image_path=image_path,
            scene_id=scene_id,
            focal_length=focal_length,
            pp_x=pp_x,
            pp_y=pp_y,
            gt_pose=gt_pose,
            calibration=calibration,
        )

    @classmethod
    def list_scenes(cls) -> List[Dict[str, str]]:
        return LocalizationService.list_scenes()


@lru_cache(maxsize=1)
def get_shared_ace_scr() -> AceSCRModel:
    """Singleton ACE SCR instance (encoder loaded once)."""
    model = AceSCRModel()
    model.load()
    return model
