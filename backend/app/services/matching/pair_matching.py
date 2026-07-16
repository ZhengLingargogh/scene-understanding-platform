"""Route pair matching to LightGlue or SIFT backend based on detector."""

from __future__ import annotations

from typing import Dict, Optional

from app.services.detectors.base import KeypointDetectionResult
from app.services.lightglue.engine import LightGlueEngine
from app.services.sift.matching import match_sift_detections

SIFT_DETECTOR_IDS = frozenset({"sift"})


def match_detection_pair(
    detection0: KeypointDetectionResult,
    detection1: KeypointDetectionResult,
    *,
    detector_model_id: Optional[str] = None,
) -> Dict[str, object]:
    detector_id = detector_model_id or detection0.backend
    if detector_id in SIFT_DETECTOR_IDS or detection0.backend == "sift":
        return match_sift_detections(detection0, detection1)
    return LightGlueEngine.get_shared().match_features_detail(detection0, detection1)
