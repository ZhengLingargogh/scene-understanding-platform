"""SALAD inference engine (DINOv2 + SALAD aggregator) without pytorch_lightning."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from app.config import settings
from app.services.salad.feature_store import _l2_normalize
from app.utils.pil_tensor import pil_to_tensor

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

_SALAD_ROOT = settings.salad_root
if str(_SALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_SALAD_ROOT))

from models import helper  # noqa: E402


class SaladVPRNetwork(nn.Module):
    """Lightweight nn.Module mirror of third_party/salad VPRModel.forward()."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = helper.get_backbone(
            "dinov2_vitb14",
            {
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
        )
        self.aggregator = helper.get_aggregator(
            "SALAD",
            {
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


class SaladEngine:
    """Shared SALAD model lifecycle for feature extraction and retrieval."""

    _shared: Optional["SaladEngine"] = None
    _shared_lock = threading.Lock()

    DESCRIPTOR_DIM = 64 * 128 + 256  # m*l + g

    def __init__(self) -> None:
        self._model: Optional[SaladVPRNetwork] = None
        self._device = torch.device(settings.salad_device)
        self._transform = self._build_transform(settings.salad_image_size)
        self._lock = threading.Lock()

    @classmethod
    def get_shared(cls) -> "SaladEngine":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def weights_path(self) -> Path:
        return settings.salad_ckpt_path

    @property
    def descriptor_dim(self) -> int:
        return self.DESCRIPTOR_DIM

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return

            ckpt_path = settings.salad_ckpt_path
            if not ckpt_path.is_file():
                raise FileNotFoundError("SALAD checkpoint not found: {}".format(ckpt_path))

            logger.info("Loading SALAD weights from %s", ckpt_path)
            model = SaladVPRNetwork()
            state_dict = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self._device)
            self._model = model
            logger.info("SALAD loaded on %s (descriptor_dim=%d)", self._device, self.DESCRIPTOR_DIM)

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            del self._model
            self._model = None
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("SALAD unloaded")

    def extract_paths(
        self,
        image_paths: Sequence[Path],
        *,
        batch_size: Optional[int] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        if not image_paths:
            raise ValueError("image_paths is empty")

        self.load()
        assert self._model is not None

        batch_size = batch_size or settings.salad_batch_size
        total = len(image_paths)
        batches: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, total, batch_size):
                batch_paths = image_paths[start : start + batch_size]
                tensors = [self._transform(self._load_image(path)) for path in batch_paths]
                batch = torch.stack(tensors, dim=0).to(self._device)

                if self._device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self._model(batch)
                else:
                    output = self._model(batch)

                batches.append(output.float().cpu().numpy())
                processed = min(start + len(batch_paths), total)
                if on_progress:
                    on_progress(processed, total, "Extracting SALAD features… {}/{}".format(processed, total))

        descriptors = np.concatenate(batches, axis=0)
        return _l2_normalize(descriptors.astype(np.float32))

    def extract_query(self, image_path: Path) -> np.ndarray:
        descriptors = self.extract_paths([Path(image_path)], batch_size=1)
        return descriptors[0]

    def top_k_similar(
        self,
        query_descriptor: np.ndarray,
        gallery_descriptors: np.ndarray,
        *,
        top_k: int = 8,
    ) -> List[Tuple[int, float]]:
        if gallery_descriptors.size == 0:
            return []

        query = query_descriptor.astype(np.float32)
        query = query / max(np.linalg.norm(query), 1e-12)
        scores = gallery_descriptors @ query
        k = min(top_k, len(scores))
        if k <= 0:
            return []

        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    @staticmethod
    def _build_transform(image_size: int) -> T.Compose:
        return T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(pil_to_tensor),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")
