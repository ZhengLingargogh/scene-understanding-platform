"""Task-oriented wrapper around the inference pipeline."""

import logging
from typing import Dict, List, Optional
from uuid import uuid4

from app.compat import UTC, datetime
from app.schemas.inference import InferenceRequest, InferenceResponse, InferenceResult
from app.schemas.pipeline import INFERENCE_PIPELINE_STAGES, PipelineStage
from app.services.inference_pipeline import run_inference_pipeline
from app.services.scene_constants import DEFAULT_SCENE_ID

logger = logging.getLogger(__name__)


class InferenceService:
    """Manages inference tasks and delegates to the plugin-based pipeline."""

    def __init__(self) -> None:
        self._tasks: Dict[str, InferenceResult] = {}
        self._results: Dict[str, dict] = {}

    def list_tasks(self) -> List[InferenceResult]:
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Optional[InferenceResult]:
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> Optional[dict]:
        return self._results.get(task_id)

    def create_task(
        self,
        payload: InferenceRequest,
        image_path: str,
        focal_length: Optional[float] = None,
        pp_x: Optional[float] = None,
        pp_y: Optional[float] = None,
        gt_pose: Optional[list] = None,
        calibration: Optional[dict] = None,
    ) -> InferenceResponse:
        task_id = str(uuid4())
        scene_id = payload.scene_id or DEFAULT_SCENE_ID
        dataset_id = payload.dataset_id
        stages = payload.pipeline_stages or list(INFERENCE_PIPELINE_STAGES)

        try:
            result = run_inference_pipeline(
                image_path=image_path,
                scene_id=scene_id,
                dataset_id=dataset_id,
                model_id=payload.model_id,
                focal_length=focal_length,
                pp_x=pp_x,
                pp_y=pp_y,
                gt_pose=gt_pose,
                calibration=calibration,
                pipeline_stages=stages,
            )
            status = "completed"
            message = "Inference completed"
        except Exception as exc:
            logger.exception("Inference failed for task %s", task_id)
            result = {"error": str(exc)}
            status = "failed"
            message = "Inference failed: {}".format(exc)

        self._results[task_id] = result
        self._tasks[task_id] = InferenceResult(
            id=task_id,
            scene_id=scene_id,
            dataset_id=dataset_id,
            model_id=payload.model_id,
            image_path=image_path,
            status=status,
            pipeline_stages=stages,
            created_at=datetime.now(UTC),
            result=result if status == "completed" else None,
            error=result.get("error") if status == "failed" else None,
        )

        return InferenceResponse(
            task_id=task_id,
            status=status,
            message=message,
            pipeline_stages=stages,
            result=result if status == "completed" else None,
        )
