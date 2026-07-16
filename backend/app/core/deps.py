"""Shared dependencies for API routes."""

from typing import Optional

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.benchmark_service import BenchmarkService
from app.services.inference_service import InferenceService
from app.services.model_service import ModelService
from app.services.scene_service import SceneService

_model_service: Optional[ModelService] = None


def get_scene_service(db: Session = Depends(get_db)) -> SceneService:
    return SceneService(db)


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def get_inference_service() -> InferenceService:
    return InferenceService()


def get_localization_service() -> InferenceService:
    """Deprecated alias."""
    return InferenceService()


def get_benchmark_service() -> BenchmarkService:
    return BenchmarkService()
