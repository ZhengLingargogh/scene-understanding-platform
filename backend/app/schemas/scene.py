from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.enums import SceneStatus


class DatasetSplitInfo(BaseModel):
    split: str
    label: str
    rgb_dir: str
    image_count: int
    dataset_id: str
    default_feature_dir: str
    calibration_dir: Optional[str] = None
    poses_dir: Optional[str] = None


class DatasetSceneInfo(BaseModel):
    scene_id: str
    name: str
    path: str
    splits: List[DatasetSplitInfo]
    registered_scene_id: Optional[UUID] = None


class DatasetCatalogResponse(BaseModel):
    id: str
    name: str
    root_path: str
    scenes: List[DatasetSceneInfo]
    is_builtin: bool = True


class SceneBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = None
    dataset_family: Optional[str] = Field(None, max_length=32)
    scene_slug: Optional[str] = Field(None, max_length=64)
    point_cloud_path: Optional[str] = Field(None, max_length=512)
    feature_index_path: Optional[str] = Field(None, max_length=512)
    reference_images_dir: Optional[str] = Field(None, max_length=512)
    train_image_count: int = Field(0, ge=0)
    test_image_count: int = Field(0, ge=0)
    status: SceneStatus = SceneStatus.unloaded


class SceneCreate(BaseModel):
    """Request body for creating a scene."""

    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = None
    dataset_family: Optional[str] = Field(None, max_length=32)
    scene_slug: Optional[str] = Field(None, max_length=64)
    point_cloud_path: Optional[str] = Field(None, max_length=512)
    feature_index_path: Optional[str] = Field(None, max_length=512)
    reference_images_dir: Optional[str] = Field(None, max_length=512)
    train_image_count: int = Field(0, ge=0)
    test_image_count: int = Field(0, ge=0)


class SceneUpdate(BaseModel):
    """Request body for partial scene updates."""

    name: Optional[str] = Field(None, min_length=1, max_length=128)
    description: Optional[str] = None
    point_cloud_path: Optional[str] = Field(None, max_length=512)
    feature_index_path: Optional[str] = Field(None, max_length=512)
    reference_images_dir: Optional[str] = Field(None, max_length=512)
    train_image_count: Optional[int] = Field(None, ge=0)
    test_image_count: Optional[int] = Field(None, ge=0)
    status: Optional[SceneStatus] = None


class SceneResponse(SceneBase):
    """API response for a scene."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
