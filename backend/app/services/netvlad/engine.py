"""NetVLAD global descriptor engine (VGG16 + NetVLAD + PCA)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from app.config import settings
from app.services.netvlad.architecture import EmbedNetPCA, NetVLAD, vgg16
from app.services.salad.feature_store import _l2_normalize
from app.utils.pil_tensor import pil_to_tensor

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class NetVLADEngine:
    _shared: Optional["NetVLADEngine"] = None
    _shared_lock = threading.Lock()

    DESCRIPTOR_DIM = 4096

    def __init__(self) -> None:
        self._model: Optional[EmbedNetPCA] = None
        self._device = torch.device(settings.netvlad_device)
        self._transform = T.Compose(
            [
                T.Resize(settings.netvlad_image_size),
                T.Lambda(pil_to_tensor),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._lock = threading.Lock()

    @classmethod
    def get_shared(cls) -> "NetVLADEngine":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def weights_path(self) -> Path:
        return settings.netvlad_ckpt_path

    @property
    def descriptor_dim(self) -> int:
        return self.DESCRIPTOR_DIM

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            ckpt_path = settings.netvlad_ckpt_path
            if not ckpt_path.is_file():
                raise FileNotFoundError("NetVLAD checkpoint not found: {}".format(ckpt_path))

            model = EmbedNetPCA(vgg16(pretrained=False), NetVLAD(num_clusters=64), dim=self.DESCRIPTOR_DIM)
            state_dict = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self._device)
            self._model = model
            logger.info("NetVLAD loaded on %s", self._device)

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            del self._model
            self._model = None
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

    def extract_paths(
        self,
        image_paths: Sequence[Path],
        *,
        batch_size: Optional[int] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        self.load()
        assert self._model is not None
        batch_size = batch_size or settings.netvlad_batch_size
        total = len(image_paths)
        batches: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, total, batch_size):
                batch_paths = image_paths[start : start + batch_size]
                tensors = [self._transform(self._load_image(path)) for path in batch_paths]
                batch = torch.stack(tensors, dim=0).to(self._device)
                output = self._model(batch)
                batches.append(output.float().cpu().numpy())
                processed = min(start + len(batch_paths), total)
                if on_progress:
                    on_progress(processed, total, "Extracting NetVLAD features… {}/{}".format(processed, total))

        return _l2_normalize(np.concatenate(batches, axis=0).astype(np.float32))

    def extract_query(self, image_path: Path) -> np.ndarray:
        return self.extract_paths([Path(image_path)], batch_size=1)[0]

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
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")
