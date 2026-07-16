"""NetVLAD image retrieval plugin."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from app.schemas.pipeline import ImageRetrievalResult, RetrievalReference
from app.services.mock_datasets import get_mock_dataset
from app.services.netvlad.engine import NetVLADEngine
from app.services.retrievers.base import Retriever
from app.services.retrievers.salad import SaladRetriever
from app.services.salad.feature_store import feature_index_exists, load_feature_index, save_feature_index
from app.services.salad.image_utils import build_image_records, list_images
from app.services.salad.paths import resolve_feature_dir, resolve_path

logger = logging.getLogger(__name__)
DEFAULT_TOP_K = 8


class NetVLADRetriever(Retriever):
    """Top-K retrieval using NetVLAD global descriptors."""

    def __init__(self, model_id: str = "netvlad") -> None:
        self._model_id = model_id
        self._engine = NetVLADEngine.get_shared()
        self._paths = SaladRetriever(model_id=model_id)

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

    def infer(
        self,
        *,
        image_path: str,
        dataset_id: str,
        top_k: int = DEFAULT_TOP_K,
        scene_id: Optional[str] = None,
    ) -> ImageRetrievalResult:
        if not self._engine.is_loaded:
            self.load()

        dataset = get_mock_dataset(dataset_id) or {}
        gallery_dir = self._paths._resolve_gallery_dir(dataset, scene_id)
        feature_dir = self._paths._resolve_feature_dir(dataset, scene_id)

        gallery_descriptors, manifest = self._load_or_build_gallery(
            gallery_dir=gallery_dir,
            feature_dir=feature_dir,
        )

        query_descriptor = self._engine.extract_query(Path(image_path))
        matches = self._engine.top_k_similar(query_descriptor, gallery_descriptors, top_k=top_k)
        images = manifest.get("images", [])

        references: List[RetrievalReference] = []
        for rank, (index, score) in enumerate(matches, start=1):
            record = images[index] if 0 <= index < len(images) else {}
            image_ref_path = record.get("path") or str(gallery_dir / record.get("filename", "unknown"))
            label = record.get("filename") or Path(image_ref_path).name
            references.append(
                RetrievalReference(
                    id="{}-rank-{}".format(self._model_id, rank),
                    label=label,
                    score=round(max(min(score, 1.0), 0.0), 4),
                    image_path=image_ref_path,
                )
            )

        return ImageRetrievalResult(
            status="completed",
            message="NetVLAD retrieval on '{}'".format(dataset.get("name", dataset_id)),
            top_k=top_k,
            dataset_id=dataset_id,
            references=references,
        )

    def _load_or_build_gallery(self, *, gallery_dir: Path, feature_dir: Optional[Path]):
        if feature_dir and feature_index_exists(feature_dir):
            descriptors, manifest = load_feature_index(feature_dir)
            if manifest.get("model_id") == self._model_id:
                return descriptors, manifest

        cache_dir = feature_dir or (gallery_dir.parent / "netvlad_features")
        if feature_index_exists(cache_dir):
            descriptors, manifest = load_feature_index(cache_dir)
            if manifest.get("model_id") == self._model_id:
                return descriptors, manifest

        logger.warning("NetVLAD index missing — extracting gallery from %s", gallery_dir)
        image_paths = list_images(gallery_dir)
        records = build_image_records(image_paths)
        descriptors = self._engine.extract_paths(image_paths)
        save_feature_index(
            cache_dir,
            model_id=self._model_id,
            image_records=records,
            descriptors=descriptors,
            descriptor_dim=self._engine.descriptor_dim,
        )
        return load_feature_index(cache_dir)
