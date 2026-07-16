from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.pipeline import BENCHMARK_PIPELINE_STAGES, PipelineStage


class BenchmarkCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=128)
    scene_id: str
    model_id: str
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = Field(
        None,
        description="Path to test dataset root (e.g. CrossLoc test split)",
    )
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    pipeline_stages: List[PipelineStage] = Field(
        default_factory=lambda: list(BENCHMARK_PIPELINE_STAGES),
        description="Benchmark stages: feature extraction, coordinate regression",
    )


class FeatureExtractionRunRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    dataset_id: str
    input_path: Optional[str] = Field(
        None,
        description="Optional; derived from dataset catalog when omitted",
    )
    output_path: str


class FeatureExtractionJobResponse(BaseModel):
    job_id: str
    type: str = "feature_extraction"
    status: str
    progress: int
    total: int
    processed: int
    message: str
    model_id: str
    dataset_id: str
    input_path: str
    output_path: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class SemanticSegmentationRunRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = "sam"
    dataset_id: str
    input_path: Optional[str] = Field(
        None,
        description="Optional; derived from dataset catalog when omitted",
    )
    output_path: str


class SemanticSegmentationJobResponse(BaseModel):
    job_id: str
    type: str = "semantic_segmentation"
    status: str
    progress: int
    total: int
    processed: int
    message: str
    model_id: str
    dataset_id: str
    input_path: str
    output_path: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class BenchmarkResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    name: str
    scene_id: str
    model_id: str
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    pipeline_stages: List[PipelineStage]
    status: str
    metrics: Optional[Dict[str, float]] = None
    created_at: datetime
    updated_at: datetime
