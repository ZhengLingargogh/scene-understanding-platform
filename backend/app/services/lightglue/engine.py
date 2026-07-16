"""LightGlue matcher engine (SuperPoint + LightGlue)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.config import settings
from app.services.detectors.base import KeypointDetectionResult

logger = logging.getLogger(__name__)

if str(settings.lightglue_root) not in sys.path:
    sys.path.insert(0, str(settings.lightglue_root))

from lightglue.lightglue import LightGlue  # noqa: E402


class LightGlueEngine:
    _shared: Optional["LightGlueEngine"] = None

    def __init__(self) -> None:
        self._matcher: Optional[LightGlue] = None
        self._device = torch.device(settings.lightglue_device)

    @classmethod
    def get_shared(cls) -> "LightGlueEngine":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    @property
    def is_loaded(self) -> bool:
        return self._matcher is not None

    @property
    def weights_path(self) -> str:
        return "third_party/lightglue (superpoint_lightglue.pth, auto-download)"

    def load(self) -> None:
        if self._matcher is None:
            self._matcher = LightGlue(features="superpoint").eval().to(self._device)
            logger.info("LightGlue matcher loaded on %s", self._device)

    def unload(self) -> None:
        if self._matcher is not None:
            del self._matcher
            self._matcher = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _scale_keypoints_to_original(
        keypoints: np.ndarray,
        processed_shape: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> List[List[float]]:
        proc_h, proc_w = processed_shape
        orig_w, orig_h = original_size
        if proc_w <= 0 or proc_h <= 0:
            return keypoints.tolist()
        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h
        scaled = keypoints.copy()
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        return scaled.tolist()

    def match_features_detail(
        self,
        detection0: KeypointDetectionResult,
        detection1: KeypointDetectionResult,
    ) -> Dict[str, object]:
        """Match precomputed keypoint features and return visualization-friendly geometry."""
        self.load()
        assert self._matcher is not None

        proc0 = detection0.processed_shape
        proc1 = detection1.processed_shape
        orig0 = detection0.original_size
        orig1 = detection1.original_size

        feats0 = {k: v.to(self._device) for k, v in detection0.feats.items()}
        feats1 = {k: v.to(self._device) for k, v in detection1.feats.items()}

        with torch.no_grad():
            pred = self._matcher({"image0": feats0, "image1": feats1})

        matches_t = pred["matches"][0]
        scores_t = pred["scores"][0]
        if matches_t.numel() == 0:
            kpts0 = feats0["keypoints"][0].cpu().numpy()
            kpts1 = feats1["keypoints"][0].cpu().numpy()
            return {
                "match_count": 0,
                "mean_score": 0.0,
                "keypoints0": self._scale_keypoints_to_original(kpts0, (proc0[1], proc0[0]), orig0),
                "keypoints1": self._scale_keypoints_to_original(kpts1, (proc1[1], proc1[0]), orig1),
                "matches": [],
                "image0_size": [orig0[0], orig0[1]],
                "image1_size": [orig1[0], orig1[1]],
            }

        matches_np = matches_t.cpu().numpy()
        mean_score = float(scores_t.mean().item()) if scores_t.numel() else 0.0
        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()

        return {
            "match_count": int(matches_np.shape[0]),
            "mean_score": mean_score,
            "keypoints0": self._scale_keypoints_to_original(kpts0, (proc0[1], proc0[0]), orig0),
            "keypoints1": self._scale_keypoints_to_original(kpts1, (proc1[1], proc1[0]), orig1),
            "matches": matches_np.tolist(),
            "image0_size": [orig0[0], orig0[1]],
            "image1_size": [orig1[0], orig1[1]],
        }

    def match_pair_detail(
        self,
        image0_path: Path,
        image1_path: Path,
        *,
        detector_model_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """Run detector + matcher on an image pair (legacy convenience)."""
        from app.services.matching.pair_matching import match_detection_pair
        from app.services.plugins.registry import get_detector

        resolved = detector_model_id or "superpoint"
        detector = get_detector(resolved)
        detection0 = detector.detect(image0_path)
        detection1 = detector.detect(image1_path)
        return match_detection_pair(detection0, detection1, detector_model_id=resolved)

    def match_pair(self, query_path: Path, reference_path: Path) -> Tuple[int, float]:
        detail = self.match_pair_detail(query_path, reference_path)
        return int(detail["match_count"]), float(detail["mean_score"])

    def match_references(
        self,
        query_path: str,
        reference_paths: List[str],
    ) -> Tuple[int, int, float]:
        total_matches = 0
        score_sum = 0.0
        pairs_with_matches = 0
        for ref in reference_paths:
            count, score = self.match_pair(Path(query_path), Path(ref))
            total_matches += count
            if count > 0:
                score_sum += score
                pairs_with_matches += 1
        avg_score = score_sum / pairs_with_matches if pairs_with_matches else 0.0
        return total_matches, pairs_with_matches, avg_score
