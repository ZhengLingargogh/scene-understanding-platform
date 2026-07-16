"""OpenCV SIFT local feature engine."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class SIFTEngine:
    _shared: Optional["SIFTEngine"] = None
    _shared_lock = threading.Lock()

    DESCRIPTOR_DIM = 128
    DEFAULT_MAX_KEYPOINTS = 2048

    def __init__(self) -> None:
        self._sift: Optional[cv2.SIFT] = None
        self._lock = threading.Lock()

    @classmethod
    def get_shared(cls) -> "SIFTEngine":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    @property
    def is_loaded(self) -> bool:
        return self._sift is not None

    @property
    def weights_path(self) -> str:
        return "opencv-contrib (built-in SIFT)"

    def load(self) -> None:
        with self._lock:
            if self._sift is not None:
                return
            self._sift = cv2.SIFT_create(nfeatures=self.DEFAULT_MAX_KEYPOINTS)
            logger.info("OpenCV SIFT loaded (nfeatures=%s)", self.DEFAULT_MAX_KEYPOINTS)

    def unload(self) -> None:
        with self._lock:
            self._sift = None

    def _read_grayscale(self, image_path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Failed to read image: {}".format(image_path))
        height, width = gray.shape[:2]
        return gray, (width, height)

    def extract_image(self, image_path: Path) -> Dict[str, np.ndarray]:
        self.load()
        assert self._sift is not None

        gray, (width, height) = self._read_grayscale(image_path)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        if not keypoints or descriptors is None:
            return {
                "keypoints": np.zeros((0, 2), dtype=np.float64),
                "scores": np.zeros((0,), dtype=np.float64),
                "descriptors": np.zeros((0, self.DESCRIPTOR_DIM), dtype=np.float32),
                "image_size": np.array([width, height], dtype=np.int32),
            }

        pts = np.array([kp.pt for kp in keypoints], dtype=np.float64)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float64)
        return {
            "keypoints": pts,
            "scores": scores,
            "descriptors": descriptors.astype(np.float32),
            "image_size": np.array([width, height], dtype=np.int32),
        }

    def extract_image_for_visualization(self, image_path: Path) -> Dict[str, np.ndarray]:
        data = self.extract_image(image_path)
        return {
            "keypoints": data["keypoints"],
            "scores": data["scores"],
        }

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
                on_progress(index, total, "Extracting SIFT features… {}/{}".format(index, total))
        return results
