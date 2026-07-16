"""Build dataset catalog enriched with registered scene IDs."""

from __future__ import annotations

from typing import Dict, List, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from app.schemas.scene import DatasetCatalogResponse, DatasetSceneInfo, DatasetSplitInfo
from app.services.dataset_catalog import get_all_dataset_families, is_builtin_family
from app.services.scene_seed import registered_scene_map


def build_dataset_catalog(db: Session) -> List[DatasetCatalogResponse]:
    reg_map = registered_scene_map(db)
    catalog: List[DatasetCatalogResponse] = []

    for family in get_all_dataset_families():
        scenes: List[DatasetSceneInfo] = []
        for scene_info in family["scenes"]:
            key: Tuple[str, str] = (family["id"], scene_info["scene_id"])
            scenes.append(
                DatasetSceneInfo(
                    scene_id=scene_info["scene_id"],
                    name=scene_info["name"],
                    path=scene_info["path"],
                    splits=[DatasetSplitInfo(**split) for split in scene_info["splits"]],
                    registered_scene_id=reg_map.get(key),
                )
            )
        catalog.append(
            DatasetCatalogResponse(
                id=family["id"],
                name=family["name"],
                root_path=family["root_path"],
                scenes=scenes,
                is_builtin=is_builtin_family(family["id"]),
            )
        )
    return catalog
