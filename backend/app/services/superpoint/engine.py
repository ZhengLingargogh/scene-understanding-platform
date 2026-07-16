"""SuperPoint local feature extraction engine (LightGlue vendored)."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.utils.pil_tensor import pil_to_tensor

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

if str(settings.lightglue_root) not in sys.path:
    sys.path.insert(0, str(settings.lightglue_root))

from lightglue.superpoint import SuperPoint  # noqa: E402


class SuperPointEngine:
    _shared: Optional["SuperPointEngine"] = None
    _shared_lock = threading.Lock()

    def __init__(self) -> None:
        self._extractor: Optional[SuperPoint] = None
        self._device = torch.device(settings.superpoint_device)
        self._lock = threading.Lock()

    @classmethod
    def get_shared(cls) -> "SuperPointEngine":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    @property
    def is_loaded(self) -> bool:
        return self._extractor is not None

    @property
    def weights_path(self) -> str:
        return "third_party/lightglue (superpoint_v1.pth, auto-download)"

    def load(self) -> None:
        with self._lock:
            if self._extractor is not None:
                return
            self._extractor = SuperPoint(max_num_keypoints=settings.superpoint_max_keypoints).eval()
            self._extractor.to(self._device)
            logger.info("SuperPoint loaded on %s", self._device)

    def unload(self) -> None:
        with self._lock:
            if self._extractor is None:
                return
            del self._extractor
            self._extractor = None
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

    def _load_tensor(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            max_side = settings.superpoint_resize
            scale = max_side / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            tensor = pil_to_tensor(img)
        return tensor

    def extract_image(self, image_path: Path) -> Dict[str, np.ndarray]:
        feats, _scale = self._extract_features(image_path)
        return feats

    def _extract_features(self, image_path: Path) -> tuple[Dict[str, np.ndarray], float]:
        """Run SuperPoint; return features and scale from original image to inference resolution."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            max_side = settings.superpoint_resize
            scale = min(1.0, max_side / max(orig_w, orig_h))
            if scale < 1.0:
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                img = img.resize((new_w, new_h), Image.BILINEAR)
            tensor = pil_to_tensor(img)

        self.load()
        assert self._extractor is not None
        image = tensor.to(self._device)
        with torch.no_grad():
            feats = self._extractor({"image": image[None]})

        keypoints = feats["keypoints"][0].cpu().numpy().astype(np.float64)
        if scale < 1.0:
            keypoints = keypoints / scale

        return {
            "keypoints": keypoints,
            "scores": feats["keypoint_scores"][0].cpu().numpy(),
            "descriptors": feats["descriptors"][0].cpu().numpy(),
        }, scale

    def extract_image_for_visualization(self, image_path: Path) -> Dict[str, np.ndarray]:
        """Keypoints/scores in original image pixel coordinates."""
        feats, _ = self._extract_features(image_path)
        return feats

    def extract_paths(
        self,
        image_paths: Sequence[Path],
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, np.ndarray]]:
        total = len(image_paths)
        results: List[Dict[str, np.ndarray]] = []
        for index, path in enumerate(image_paths, start=1):
            results.append(self.extract_image(path))
            if on_progress:
                on_progress(index, total, "Extracting SuperPoint features… {}/{}".format(index, total))
        return results

    def extract_tensor(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.load()
        assert self._extractor is not None
        if image_tensor.ndim == 3:
            image_tensor = image_tensor[None]
        with torch.no_grad():
            return self._extractor({"image": image_tensor.to(self._device)})

    def load_tensor_from_path(self, image_path: Path) -> torch.Tensor:
        return self._load_tensor(image_path)
