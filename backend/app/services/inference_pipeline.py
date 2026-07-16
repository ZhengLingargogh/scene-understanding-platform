"""Orchestrates single-image inference via plugin interfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.schemas.pipeline import (
    INFERENCE_PIPELINE_STAGES,
    ImageMatchingResult,
    ImageRetrievalResult,
    PipelineResults,
    PipelineStage,
)
from app.services.plugins import get_matcher, get_retriever
from app.services.retrievers import DEFAULT_TOP_K


def _resolve_retriever_model_id(model_id: str) -> str:
    from app.services.plugins.registry import RETRIEVER_REGISTRY, DEFAULT_RETRIEVER_ID

    if model_id in RETRIEVER_REGISTRY:
        return model_id
    return DEFAULT_RETRIEVER_ID


def _resolve_matcher_model_id(matcher_model_id: Optional[str], fallback_model_id: str) -> str:
    from app.services.plugins.registry import MATCHER_REGISTRY, DEFAULT_MATCHER_ID

    if matcher_model_id and matcher_model_id in MATCHER_REGISTRY:
        return matcher_model_id
    if fallback_model_id in MATCHER_REGISTRY:
        return fallback_model_id
    return DEFAULT_MATCHER_ID


def _resolve_detector_model_id(detector_model_id: Optional[str]) -> str:
    from app.services.plugins.registry import DETECTOR_REGISTRY, DEFAULT_DETECTOR_ID

    if detector_model_id and detector_model_id in DETECTOR_REGISTRY:
        return detector_model_id
    return DEFAULT_DETECTOR_ID


def run_inference_pipeline(
    image_path: str,
    scene_id: str,
    dataset_id: str,
    model_id: str = "salad",
    focal_length: Optional[float] = None,
    pp_x: Optional[float] = None,
    pp_y: Optional[float] = None,
    gt_pose: Optional[list] = None,
    calibration: Optional[dict] = None,
    pipeline_stages: Optional[List[PipelineStage]] = None,
    scr_model_id: Optional[str] = None,
    matcher_model_id: Optional[str] = None,
    detector_model_id: Optional[str] = None,
    reference_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inference pipeline (plugin-based):
    Retriever.infer → Matcher.infer / infer_pair
    """
    stages = pipeline_stages or list(INFERENCE_PIPELINE_STAGES)
    stage_set = set(stages)
    retriever_id = _resolve_retriever_model_id(model_id)
    matcher_id = _resolve_matcher_model_id(matcher_model_id, model_id)
    detector_id = _resolve_detector_model_id(detector_model_id)

    retrieval_result: Optional[ImageRetrievalResult] = None
    matching_result: Optional[ImageMatchingResult] = None

    if PipelineStage.image_retrieval in stage_set:
        retriever = get_retriever(retriever_id)
        retrieval_result = retriever.infer(
            image_path=image_path,
            dataset_id=dataset_id,
            top_k=DEFAULT_TOP_K,
            scene_id=scene_id,
        )

    if PipelineStage.image_matching in stage_set:
        matcher = get_matcher(matcher_id)
        if reference_image_path:
            matching_result = matcher.infer_pair(
                image0_path=image_path,
                image1_path=reference_image_path,
                scene_id=scene_id,
                detector_model_id=detector_id,
            )
        else:
            refs = retrieval_result.references if retrieval_result else []
            matching_result = matcher.infer(
                query_image_path=image_path,
                references=refs,
                scene_id=scene_id,
            )

    pipeline = PipelineResults(
        image_retrieval=retrieval_result,
        image_matching=matching_result,
    )

    return {
        "scene_id": scene_id,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "retriever_model_id": retriever_id,
        "matcher_model_id": matcher_id,
        "detector_model_id": detector_id,
        "pipeline_stages": [s.value for s in stages],
        "pipeline": pipeline.model_dump(exclude_none=True),
    }
