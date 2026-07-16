"""Seed and sync SQLite scene rows from the dataset catalog."""

from __future__ import annotations

from typing import Dict, Tuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.compat import UTC, datetime
from app.db.enums import SceneStatus
from app.db.models import Scene
from app.services.dataset_catalog import get_all_dataset_families


def _scene_key(family_id: str, scene_slug: str) -> Tuple[str, str]:
    return family_id, scene_slug


def seed_catalog_scenes(db: Session) -> int:
    """Upsert catalog scenes. Returns number of scenes created."""
    created = 0
    now = datetime.now(UTC)

    existing_rows = db.scalars(select(Scene)).all()
    by_key: Dict[Tuple[str, str], Scene] = {}
    by_name: Dict[str, Scene] = {}
    for row in existing_rows:
        if row.dataset_family and row.scene_slug:
            by_key[_scene_key(row.dataset_family, row.scene_slug)] = row
        by_name[row.name] = row

    for family in get_all_dataset_families():
        for scene_info in family["scenes"]:
            scene_slug = scene_info["scene_id"]
            train_split = next(s for s in scene_info["splits"] if s["split"] == "train")
            test_split = next(s for s in scene_info["splits"] if s["split"] == "test")
            name = f"{family['name']} · {scene_info['name']}"
            key = _scene_key(family["id"], scene_slug)
            row = by_key.get(key) or by_name.get(name)

            if row is None:
                row = Scene(
                    name=name,
                    description=f"{family['id']} 数据集 · {scene_slug}",
                    dataset_family=family["id"],
                    scene_slug=scene_slug,
                    point_cloud_path=None,
                    feature_index_path=train_split["default_feature_dir"],
                    reference_images_dir=train_split["rgb_dir"],
                    train_image_count=train_split["image_count"],
                    test_image_count=test_split["image_count"],
                    status=SceneStatus.unloaded,
                    created_at=now,
                    updated_at=now,
                )
                db.add(row)
                by_key[key] = row
                created += 1
            else:
                row.name = name
                row.description = f"{family['id']} 数据集 · {scene_slug}"
                row.dataset_family = family["id"]
                row.scene_slug = scene_slug
                row.feature_index_path = train_split["default_feature_dir"]
                row.reference_images_dir = train_split["rgb_dir"]
                row.train_image_count = train_split["image_count"]
                row.test_image_count = test_split["image_count"]
                row.updated_at = now

    db.commit()
    return created


def registered_scene_map(db: Session) -> Dict[Tuple[str, str], UUID]:
    """Map (dataset_family, scene_slug) → SQLite scene UUID."""
    rows = db.scalars(
        select(Scene).where(Scene.dataset_family.is_not(None), Scene.scene_slug.is_not(None))
    ).all()
    result: Dict[Tuple[str, str], UUID] = {}
    for row in rows:
        if row.dataset_family and row.scene_slug:
            result[_scene_key(row.dataset_family, row.scene_slug)] = row.id
    return result
