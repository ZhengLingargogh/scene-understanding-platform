"""Deprecated: use app.schemas.inference instead."""

from app.schemas.inference import (  # noqa: F401
    InferenceRequest as LocalizationRequest,
    InferenceResponse as LocalizationResponse,
    InferenceResult as LocalizationResult,
)

__all__ = ["LocalizationRequest", "LocalizationResponse", "LocalizationResult"]
