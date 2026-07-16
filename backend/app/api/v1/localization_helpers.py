"""Backward-compatible alias for inference helpers."""

from app.api.v1.inference_helpers import resolve_inference_inputs as resolve_localization_inputs

__all__ = ["resolve_localization_inputs"]
