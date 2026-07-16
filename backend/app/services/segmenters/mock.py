"""Mock interactive segmenter — elliptical mask around the point prompt."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.segmenters.base import Segmenter, SegmentationResult


class MockSegmenter(Segmenter):
    def __init__(self, model_id: str = "mock-segmenter") -> None:
        self._model_id = model_id
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def segment_point(
        self,
        *,
        image_path: Path,
        point_x: float,
        point_y: float,
        image_width: int,
        image_height: int,
    ) -> SegmentationResult:
        if not self._loaded:
            self.load()

        width = max(1, int(image_width))
        height = max(1, int(image_height))
        cx = float(np.clip(point_x, 0, width - 1))
        cy = float(np.clip(point_y, 0, height - 1))

        rng = np.random.default_rng(
            int(cx) * 73856093 ^ int(cy) * 19349663 ^ width * 83492791 ^ height
        )
        base_radius = min(width, height) * float(rng.uniform(0.12, 0.22))
        rx = int(max(12, base_radius))
        ry = int(max(10, base_radius * float(rng.uniform(0.75, 1.05))))

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (rx, ry), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0)
        mask = (mask > 0.35).astype(np.uint8)

        contour = _largest_contour(mask)
        area_ratio = float(mask.sum()) / float(width * height)
        score = float(np.clip(0.55 + area_ratio * 2.5 + rng.uniform(0, 0.08), 0.0, 0.99))

        return SegmentationResult(
            mask=mask,
            contour=contour,
            score=round(score, 4),
            backend=self._model_id,
            message="Mock elliptical mask at ({:.0f}, {:.0f})".format(cx, cy),
        )


def _largest_contour(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
