"""SALAD image retrieval plugin (cosine similarity on precomputed descriptors)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from app.schemas.pipeline import ImageRetrievalResult, RetrievalReference
from app.services.mock_datasets import get_mock_dataset
from app.services.retrievers.base import Retriever
from app.services.salad.engine import SaladEngine
from app.services.salad.feature_store import feature_index_exists, load_feature_index, save_feature_index
from app.services.salad.image_utils import build_image_records, list_images
from app.services.salad.paths import resolve_feature_dir, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 8


class SaladRetriever(Retriever):
    """Top-K retrieval using SALAD global descriptors."""

    def __init__(self, model_id: str = "salad") -> None:
        self._model_id = model_id
        self._engine = SaladEngine.get_shared()

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
        gallery_dir = self._resolve_gallery_dir(dataset, scene_id)
        feature_dir = self._resolve_feature_dir(dataset, scene_id)

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

        dataset_name = dataset.get("name", dataset_id)
        return ImageRetrievalResult(
            status="completed",
            message="SALAD retrieval on '{}' (gallery: {}, features: {})".format(
                dataset_name,
                gallery_dir,
                feature_dir or gallery_dir,
            ),
            top_k=top_k,
            dataset_id=dataset_id,
            references=references,
        )

    def _resolve_gallery_dir(self, dataset: dict, scene_id: Optional[str]) -> Path:
        scene_paths = self._scene_paths(scene_id)
        if scene_paths and scene_paths.get("reference_images_dir"):
            return resolve_path(scene_paths["reference_images_dir"])

        gallery = (
            dataset.get("gallery_rgb_dir")
            or dataset.get("reference_rgb_dir")
            or dataset.get("query_rgb_dir")
        )
        if gallery:
            return resolve_path(str(gallery))

        raise ValueError(
            "No reference image directory for dataset '{}' (scene_id={}). "
            "Set scene.reference_images_dir or dataset.reference_rgb_dir.".format(
                dataset.get("id", "unknown"),
                scene_id,
            )
        )

    def _resolve_feature_dir(self, dataset: dict, scene_id: Optional[str]) -> Optional[Path]:
        scene_paths = self._scene_paths(scene_id)
        scene_feature = scene_paths.get("feature_index_path") if scene_paths else None
        dataset_feature = dataset.get("default_feature_dir")
        return resolve_feature_dir(
            scene_feature_index_path=scene_feature,
            dataset_feature_dir=str(dataset_feature) if dataset_feature else None,
        )

    def _load_or_build_gallery(self, *, gallery_dir: Path, feature_dir: Optional[Path]):
        if feature_dir and feature_index_exists(feature_dir):
            return load_feature_index(feature_dir)

        cache_dir = feature_dir or (gallery_dir.parent / "salad_features")
        if feature_index_exists(cache_dir):
            return load_feature_index(cache_dir)

        logger.warning(
            "SALAD feature index missing at %s — extracting gallery on-the-fly from %s",
            cache_dir,
            gallery_dir,
        )
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

    @staticmethod
    def _scene_paths(scene_id: Optional[str]) -> dict:
        if not scene_id:
            return {}

        try:
            from app.db.session import SessionLocal
            from app.services.scene_service import SceneService

            with SessionLocal() as db:
                service = SceneService(db)
                scene = service.get_scene(scene_id)
                if scene is None:
                    for row in service.list_scenes():
                        if row.name.lower() == str(scene_id).lower():
                            scene = row
                            break
                if scene is None:
                    return {}
                return {
                    "reference_images_dir": scene.reference_images_dir,
                    "feature_index_path": scene.feature_index_path,
                }
        except Exception as exc:
            logger.debug("Could not resolve scene paths for %s: %s", scene_id, exc)
            return {}
