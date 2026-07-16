"""Schemas for custom dataset registration."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class RegisterDatasetRequest(BaseModel):
    root_path: str = Field(..., min_length=1, description="数据集根目录绝对路径")


class DatasetSplitScanInfo(BaseModel):
    split: str
    label: str
    rgb_dir: str
    calibration_dir: str
    poses_dir: str
    image_count: int
    dataset_id: str
    default_feature_dir: str


class DatasetSceneScanInfo(BaseModel):
    scene_id: str
    name: str
    path: str
    splits: List[DatasetSplitScanInfo]


class DatasetScanPreview(BaseModel):
    family_id: str
    name: str
    root_path: str
    scenes: List[DatasetSceneScanInfo]
    warnings: List[str] = Field(default_factory=list)
    valid: bool = False


class RegisterDatasetResponse(BaseModel):
    family_id: str
    name: str
    root_path: str
    scene_count: int
    warnings: List[str] = Field(default_factory=list)
    registered_at: datetime


class CustomDatasetSummary(BaseModel):
    family_id: str
    name: str
    root_path: str
    scene_count: int
    registered_at: datetime


class BrowseEntry(BaseModel):
    name: str
    path: str
    is_directory: bool


class BrowseResponse(BaseModel):
    path: str
    parent_path: Optional[str] = None
    entries: List[BrowseEntry]
