"""Pydantic schemas for interactive segmentation API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SegmentationSessionResponse(BaseModel):
    session_id: str
    model_id: str
    image_width: int
    image_height: int
    backend: str
    message: str = ""


class SegmentationPredictResponse(BaseModel):
    session_id: str
    model_id: str
    point: List[float] = Field(description="Prompt point [x, y] in original image pixels")
    image_width: int
    image_height: int
    mask: List[int] = Field(description="Flattened binary mask (row-major, 0/1)")
    contour: List[List[float]] = Field(default_factory=list, description="Mask boundary polygon")
    score: float = 0.0
    backend: str
    message: str = ""
