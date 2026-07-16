from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.pipeline import DEFAULT_PIPELINE_STAGES, PipelineStage


class ModelBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=128)
    version: str = Field(..., min_length=1, max_length=64)
    description: Optional[str] = None
    capabilities: List[PipelineStage] = Field(
        default_factory=lambda: list(DEFAULT_PIPELINE_STAGES),
        description="Pipeline stages this model supports",
    )


class ModelCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=128)
    version: str = Field(..., min_length=1, max_length=64)
    description: Optional[str] = None
    capabilities: List[PipelineStage] = Field(default_factory=lambda: list(DEFAULT_PIPELINE_STAGES))


class ModelUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: Optional[str] = Field(None, min_length=1, max_length=128)
    version: Optional[str] = Field(None, min_length=1, max_length=64)
    description: Optional[str] = None
    capabilities: Optional[List[PipelineStage]] = None


class ModelResponse(ModelBase):
    id: str
    status: str
    weights_path: Optional[str] = None
    plugin_loaded: Optional[bool] = None
    created_at: datetime
    updated_at: datetime


class ModelRuntimeStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    status: str
    plugin_loaded: bool = False
    weights_path: Optional[str] = None
    capabilities: List[PipelineStage] = Field(default_factory=list)
