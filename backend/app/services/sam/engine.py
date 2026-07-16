"""Original Segment Anything (SAM) image segmentation engine."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


def _ensure_sam_on_path() -> None:
    root = Path(settings.sam_root)
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


class SAMEngine:
    """Shared SAM image predictor with cached embeddings for interactive prompts."""

    _shared: Optional["SAMEngine"] = None
    _shared_lock = threading.Lock()

    def __init__(self) -> None:
        self._predictor = None
        self._mask_generator = None
        self._loaded = False
        self._prepared_path: Optional[str] = None
        self._lock = threading.Lock()
        self._device = settings.sam_device

    @classmethod
    def get_shared(cls) -> "SAMEngine":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._predictor is not None

    @property
    def weights_path(self) -> Path:
        return Path(settings.sam_ckpt_path)

    def load(self) -> None:
        with self._lock:
            if self._loaded and self._predictor is not None:
                return

            ckpt = Path(settings.sam_ckpt_path)
            if not ckpt.is_file():
                raise FileNotFoundError(
                    "SAM checkpoint not found: {}. "
                    "Place sam_vit_b_01ec64.pth under models/sam/.".format(ckpt)
                )

            _ensure_sam_on_path()
            from segment_anything import SamPredictor, sam_model_registry

            model_type = settings.sam_model_type
            if model_type not in sam_model_registry:
                raise ValueError(
                    "Unknown SAM model type '{}'. Expected one of: {}".format(
                        model_type, ", ".join(sorted(sam_model_registry.keys()))
                    )
                )

            sam_model = sam_model_registry[model_type](checkpoint=str(ckpt))
            sam_model.to(device=self._device)
            sam_model.eval()
            self._predictor = SamPredictor(sam_model)
            self._loaded = True
            self._prepared_path = None
            logger.info(
                "SAM loaded from %s (type=%s, device=%s)",
                ckpt,
                model_type,
                self._device,
            )

    def unload(self) -> None:
        with self._lock:
            self._predictor = None
            self._mask_generator = None
            self._loaded = False
            self._prepared_path = None
            try:
                import torch

                if "cuda" in self._device:
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def prepare_image(self, image_path: Path) -> None:
        """Encode image once so subsequent point prompts are fast."""
        self.load()
        assert self._predictor is not None
        path = str(Path(image_path).resolve())
        with self._lock:
            if self._prepared_path == path and self._predictor.is_image_set:
                return
            with Image.open(path) as img:
                rgb = np.asarray(img.convert("RGB"))
            self._predictor.set_image(rgb)
            self._prepared_path = path
            logger.info("SAM image embedding ready for %s", path)

    def segment_point(
        self,
        image_path: Path,
        point_x: float,
        point_y: float,
    ) -> Dict[str, object]:
        """Run point-prompt segmentation; returns mask (H,W uint8), score, contour."""
        path = Path(image_path)
        resolved = str(path.resolve())
        self.prepare_image(path)

        assert self._predictor is not None
        with self._lock:
            if self._prepared_path != resolved:
                with Image.open(resolved) as img:
                    rgb = np.asarray(img.convert("RGB"))
                self._predictor.set_image(rgb)
                self._prepared_path = resolved

            masks, scores, _logits = self._predictor.predict(
                point_coords=np.array([[float(point_x), float(point_y)]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            with Image.open(resolved) as img:
                width, height = img.size
            return {
                "mask": np.zeros((height, width), dtype=np.uint8),
                "score": 0.0,
                "contour": [],
                "message": "SAM produced no mask at ({:.0f}, {:.0f})".format(point_x, point_y),
            }

        best_idx = int(np.argmax(np.asarray(scores)))
        mask = np.ascontiguousarray(np.asarray(masks[best_idx]) > 0, dtype=np.uint8)
        score = float(np.asarray(scores)[best_idx])
        contour = _largest_contour(mask)

        return {
            "mask": mask,
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "contour": contour,
            "message": "SAM mask at ({:.0f}, {:.0f})".format(point_x, point_y),
        }

    def segment_largest_auto(self, image_path: Path) -> Dict[str, object]:
        """Generate automatic masks and return the largest region as a binary mask."""
        path = Path(image_path)
        resolved = str(path.resolve())
        self.load()
        assert self._predictor is not None

        with Image.open(resolved) as img:
            rgb = np.asarray(img.convert("RGB"))
            height, width = rgb.shape[:2]

        with self._lock:
            if self._mask_generator is None:
                _ensure_sam_on_path()
                from segment_anything import SamAutomaticMaskGenerator

                self._mask_generator = SamAutomaticMaskGenerator(
                    self._predictor.model,
                    points_per_side=16,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=0,
                )
            annotations = self._mask_generator.generate(rgb)

        if not annotations:
            return {
                "mask": np.zeros((height, width), dtype=np.uint8),
                "score": 0.0,
                "mask_count": 0,
                "message": f"SAM automatic masks empty for {path.name}",
            }

        best = max(annotations, key=lambda item: int(item.get("area", 0)))
        segmentation = np.asarray(best["segmentation"], dtype=np.uint8)
        mask = np.ascontiguousarray(segmentation > 0, dtype=np.uint8)
        score = float(best.get("predicted_iou", best.get("stability_score", 0.0)))

        return {
            "mask": mask,
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "mask_count": len(annotations),
            "message": f"SAM largest auto mask for {path.name}",
        }


def _largest_contour(mask: np.ndarray) -> List[List[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
