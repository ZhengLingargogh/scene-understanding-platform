import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.inference_helpers import resolve_inference_inputs
from app.core.deps import get_inference_service
from app.schemas.inference import InferenceRequest, InferenceResponse, InferenceResult
from app.schemas.pipeline import INFERENCE_PIPELINE_STAGES, PipelineStage
from app.services.inference_pipeline import run_inference_pipeline
from app.services.inference_service import InferenceService
from app.services.scene_constants import DEFAULT_SCENE_ID
from app.services.mock_datasets import list_mock_datasets
from app.services.plugins import registry

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_pipeline_stages(raw: Optional[str]) -> List[PipelineStage]:
    if not raw or raw.strip().lower() in {"", "string", "null"}:
        return list(INFERENCE_PIPELINE_STAGES)
    try:
        values = json.loads(raw)
        if not isinstance(values, list):
            raise ValueError("pipeline_stages must be a JSON array")
        return [PipelineStage(v) for v in values]
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid pipeline_stages: {}".format(exc),
        ) from exc


@router.get("/datasets")
async def list_inference_datasets():
    """Datasets available for image retrieval."""
    return list_mock_datasets()


@router.get("/plugins")
async def list_inference_plugins():
    """Registered CV plugin implementations (for debugging / docs)."""
    return registry.list_plugins()



@router.get("/pipeline-stages")
async def list_pipeline_stages():
    """List inference pipeline stages."""
    from app.schemas.pipeline import PIPELINE_STAGE_LABELS

    return [
        {"stage": stage.value, "label": PIPELINE_STAGE_LABELS[stage]}
        for stage in INFERENCE_PIPELINE_STAGES
    ]


@router.post("/feature-visualization", status_code=status.HTTP_200_OK)
async def run_feature_visualization(
    model_id: str = Form(default="superpoint"),
    image: UploadFile = File(...),
):
    """
    Extract keypoints + confidence scores for feature visualization heatmap.
    Supports local keypoint models: superpoint, sift.
    """
    from app.api.v1.inference_helpers import save_upload
    from app.services.feature_visualization import run_feature_visualization as run_viz

    if not image.filename:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="image is required")

    try:
        image_path = await save_upload(image, "images")
        return run_viz(str(image_path), model_id=model_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Feature visualization failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature visualization failed: {}".format(exc),
        ) from exc


@router.post("/interactive-segmentation/session", status_code=status.HTTP_200_OK)
async def create_interactive_segmentation_session(
    model_id: str = Form(default="sam"),
    image: UploadFile = File(...),
):
    """
    Upload a query image and start an interactive segmentation session.
    Subsequent point prompts use ``/interactive-segmentation/predict``.
    """
    from app.api.v1.inference_helpers import save_upload
    from app.services.interactive_segmentation import get_interactive_segmentation_service
    from app.services.plugins.registry import SEGMENTER_REGISTRY

    if not image.filename:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="image is required")
    if model_id not in SEGMENTER_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unknown segmentation model_id '{}'".format(model_id),
        )

    try:
        image_path = str(await save_upload(image, "images"))
        service = get_interactive_segmentation_service()
        return service.create_session(image_path=image_path, model_id=model_id)
    except Exception as exc:
        logger.exception("Interactive segmentation session failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create segmentation session: {}".format(exc),
        ) from exc


@router.post("/interactive-segmentation/predict", status_code=status.HTTP_200_OK)
async def predict_interactive_segmentation(
    session_id: str = Form(...),
    point_x: float = Form(...),
    point_y: float = Form(...),
):
    """Send a foreground point prompt (x, y) and receive a binary mask + contour."""
    from app.services.interactive_segmentation import get_interactive_segmentation_service

    try:
        service = get_interactive_segmentation_service()
        return service.predict(session_id=session_id, point_x=point_x, point_y=point_y)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Interactive segmentation predict failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Segmentation predict failed: {}".format(exc),
        ) from exc


@router.post("/run", response_model=dict, status_code=status.HTTP_200_OK)
async def run_inference_sync(
    scene_id: str = Form(default=DEFAULT_SCENE_ID),
    dataset_id: str = Form(default="crossloc-nature-test"),
    model_id: str = Form(default="salad"),
    image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(default=None),
    calibration_file: Optional[UploadFile] = File(default=None),
    gt_pose_file: Optional[UploadFile] = File(default=None),
    focal_length: Optional[float] = Form(default=None),
    pp_x: Optional[float] = Form(default=None),
    pp_y: Optional[float] = Form(default=None),
    gt_pose: Optional[str] = Form(default=None),
    pipeline_stages: Optional[str] = Form(default=None),
    matcher_model_id: Optional[str] = Form(default=None),
    detector_model_id: Optional[str] = Form(default="superpoint"),
):
    """
    Single-image inference via plugins:
    Retriever.infer → Matcher.infer / infer_pair

    Optional ``reference_image`` enables direct pair matching (detector → matcher on two uploads).
    """
    stages = _parse_pipeline_stages(pipeline_stages)

    try:
        image_path, calibration, parsed_gt, _meta = await resolve_inference_inputs(
            image=image,
            calibration_file=calibration_file,
            gt_pose_file=gt_pose_file,
            focal_length=focal_length,
            pp_x=pp_x,
            pp_y=pp_y,
            gt_pose=gt_pose,
        )

        reference_image_path = None
        if reference_image is not None and reference_image.filename:
            from app.api.v1.inference_helpers import save_upload

            reference_image_path = str(await save_upload(reference_image, "images"))

        return run_inference_pipeline(
            image_path=image_path,
            scene_id=scene_id,
            dataset_id=dataset_id,
            model_id=model_id,
            calibration=calibration,
            gt_pose=parsed_gt,
            pipeline_stages=stages,
            matcher_model_id=matcher_model_id,
            detector_model_id=detector_model_id,
            reference_image_path=reference_image_path,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed: {}. Check server logs; common fix: pip install 'numpy<2'".format(exc),
        ) from exc


@router.get("", response_model=List[InferenceResult])
async def list_inference_tasks(
    service: InferenceService = Depends(get_inference_service),
):
    return service.list_tasks()


@router.post("", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
async def create_inference_task(
    scene_id: str = Form(default=DEFAULT_SCENE_ID),
    dataset_id: str = Form(default="crossloc-nature-test"),
    model_id: str = Form(default="salad"),
    image: UploadFile = File(...),
    calibration_file: Optional[UploadFile] = File(default=None),
    gt_pose_file: Optional[UploadFile] = File(default=None),
    focal_length: Optional[float] = Form(default=None),
    pp_x: Optional[float] = Form(default=None),
    pp_y: Optional[float] = Form(default=None),
    gt_pose: Optional[str] = Form(default=None),
    pipeline_stages: Optional[str] = Form(default=None),
    service: InferenceService = Depends(get_inference_service),
):
    """Upload query image and run inference pipeline (task record)."""
    stages = _parse_pipeline_stages(pipeline_stages)

    image_path, calibration, parsed_gt, _meta = await resolve_inference_inputs(
        image=image,
        calibration_file=calibration_file,
        gt_pose_file=gt_pose_file,
        focal_length=focal_length,
        pp_x=pp_x,
        pp_y=pp_y,
        gt_pose=gt_pose,
    )

    payload = InferenceRequest(
        scene_id=scene_id,
        dataset_id=dataset_id,
        model_id=model_id,
        pipeline_stages=stages,
    )
    return service.create_task(
        payload,
        image_path,
        calibration=calibration,
        gt_pose=parsed_gt,
    )


@router.get("/{task_id}", response_model=InferenceResult)
async def get_inference_task(
    task_id: str,
    service: InferenceService = Depends(get_inference_service),
):
    task = service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return task
