"""LightGlue image matching plugin (detector + LightGlue matcher)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.schemas.pipeline import ImageMatchingResult, RetrievalReference
from app.services.lightglue.engine import LightGlueEngine
from app.services.matchers.base import Matcher
from app.services.matching.pair_matching import match_detection_pair


class LightGlueMatcher(Matcher):
    def __init__(self, model_id: str = "lightglue") -> None:
        self._model_id = model_id
        self._engine = LightGlueEngine.get_shared()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    def load(self) -> None:
        self._engine.load()

    def unload(self) -> None:
        self._engine.unload()

    def infer_pair(
        self,
        *,
        image0_path: str,
        image1_path: str,
        scene_id: Optional[str] = None,
        detector_model_id: Optional[str] = None,
    ) -> ImageMatchingResult:
        resolved_detector = detector_model_id or "superpoint"
        if resolved_detector != "sift" and not self._engine.is_loaded:
            self.load()
        from app.services.plugins.registry import get_detector

        detector = get_detector(resolved_detector)
        detection0 = detector.detect(Path(image0_path))
        detection1 = detector.detect(Path(image1_path))
        detail = match_detection_pair(
            detection0,
            detection1,
            detector_model_id=resolved_detector,
        )
        matcher_label = "BFMatcher" if resolved_detector == "sift" else "LightGlue"
        match_count = int(detail["match_count"])
        mean_score = float(detail["mean_score"])
        inlier_ratio = min(1.0, match_count / max(len(detail["keypoints0"]), 1))  # type: ignore[arg-type]

        return ImageMatchingResult(
            status="completed",
            message="{} + {} matched {} keypoint pairs (avg score {:.3f})".format(
                resolved_detector,
                matcher_label,
                match_count,
                mean_score,
            ),
            match_count=match_count,
            inlier_count=match_count,
            inlier_ratio=round(inlier_ratio, 4),
            keypoints0=detail["keypoints0"],  # type: ignore[arg-type]
            keypoints1=detail["keypoints1"],  # type: ignore[arg-type]
            matches=detail["matches"],  # type: ignore[arg-type]
            image0_path=image0_path,
            image1_path=image1_path,
            image0_size=detail["image0_size"],  # type: ignore[arg-type]
            image1_size=detail["image1_size"],  # type: ignore[arg-type]
        )

    def infer(
        self,
        *,
        query_image_path: str,
        references: List[RetrievalReference],
        scene_id: Optional[str] = None,
        detector_model_id: Optional[str] = None,
    ) -> ImageMatchingResult:
        ref_paths = [ref.image_path for ref in references if ref.image_path]
        if not ref_paths:
            return ImageMatchingResult(
                status="failed",
                message="No reference image paths for LightGlue matching",
                match_count=0,
                inlier_count=0,
                inlier_ratio=0.0,
            )

        if len(ref_paths) == 1:
            return self.infer_pair(
                image0_path=query_image_path,
                image1_path=ref_paths[0],
                scene_id=scene_id,
                detector_model_id=detector_model_id,
            )

        total_matches = 0
        score_sum = 0.0
        pairs_matched = 0
        for ref in ref_paths:
            from app.services.plugins.registry import get_detector

            detector = get_detector(detector_model_id or "superpoint")
            det0 = detector.detect(Path(query_image_path))
            det1 = detector.detect(Path(ref))
            detail = match_detection_pair(
                det0,
                det1,
                detector_model_id=detector_model_id,
            )
            count = int(detail["match_count"])
            total_matches += count
            if count > 0:
                score_sum += float(detail["mean_score"])
                pairs_matched += 1
        avg_score = score_sum / pairs_matched if pairs_matched else 0.0
        inlier_ratio = min(1.0, total_matches / max(len(ref_paths) * 50, 1))

        return ImageMatchingResult(
            status="completed",
            message="LightGlue matched {} pairs across {} references (avg score {:.3f})".format(
                pairs_matched,
                len(ref_paths),
                avg_score,
            ),
            match_count=total_matches,
            inlier_count=total_matches,
            inlier_ratio=round(inlier_ratio, 4),
            image0_path=query_image_path,
        )
