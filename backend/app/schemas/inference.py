from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.pipeline import INFERENCE_PIPELINE_STAGES, PipelineResults, PipelineStage


class InferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    scene_id: str = Field(default="nature", description="Scene key (ACE head or DB scene)")
    dataset_id: str = Field(default="crossloc-nature-test", description="Dataset for image retrieval")
    model_id: str = Field(default="ace", description="Registered model identifier")
    image_path: Optional[str] = None
    pipeline_stages: List[PipelineStage] = Field(
        default_factory=lambda: list(INFERENCE_PIPELINE_STAGES),
        description="Inference stages: retrieval, matching, coordinate regression",
    )


class InferenceResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    scene_id: str
    dataset_id: str
    model_id: str
    image_path: str
    status: str
    pipeline_stages: List[PipelineStage]
    created_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class InferenceResponse(BaseModel):
    task_id: str
    status: str
    message: str = Field(default="Inference task accepted")
    pipeline_stages: List[PipelineStage] = Field(default_factory=lambda: list(INFERENCE_PIPELINE_STAGES))
    result: Optional[Dict[str, Any]] = None
