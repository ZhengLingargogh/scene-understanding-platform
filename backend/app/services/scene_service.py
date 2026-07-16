from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.compat import UTC, datetime
from app.db.enums import SceneStatus
from app.db.models import Scene
from app.schemas.scene import SceneCreate, SceneResponse, SceneUpdate


class SceneService:
    """Scene CRUD backed by SQLAlchemy."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def list_scenes(self) -> List[SceneResponse]:
        rows = self._db.scalars(select(Scene).order_by(Scene.created_at.desc())).all()
        return [SceneResponse.model_validate(row) for row in rows]

    def get_scene(self, scene_id: str | UUID) -> Optional[SceneResponse]:
        row = self._db.get(Scene, UUID(str(scene_id)))
        if row is None:
            return None
        return SceneResponse.model_validate(row)

    def create_scene(self, payload: SceneCreate) -> SceneResponse:
        now = datetime.now(UTC)
        row = Scene(
            name=payload.name,
            description=payload.description,
            dataset_family=payload.dataset_family,
            scene_slug=payload.scene_slug,
            point_cloud_path=payload.point_cloud_path,
            feature_index_path=payload.feature_index_path,
            reference_images_dir=payload.reference_images_dir,
            train_image_count=payload.train_image_count,
            test_image_count=payload.test_image_count,
            status=SceneStatus.unloaded,
            created_at=now,
            updated_at=now,
        )
        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)
        return SceneResponse.model_validate(row)

    def update_scene(self, scene_id: str | UUID, payload: SceneUpdate) -> Optional[SceneResponse]:
        row = self._db.get(Scene, UUID(str(scene_id)))
        if row is None:
            return None

        updates = payload.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(row, key, value)
        row.updated_at = datetime.now(UTC)

        self._db.commit()
        self._db.refresh(row)
        return SceneResponse.model_validate(row)

    def delete_scenes_by_family(self, family_id: str) -> int:
        rows = self._db.scalars(select(Scene).where(Scene.dataset_family == family_id)).all()
        for row in rows:
            self._db.delete(row)
        if rows:
            self._db.commit()
        return len(rows)

    def delete_scene(self, scene_id: str | UUID) -> bool:
        row = self._db.get(Scene, UUID(str(scene_id)))
        if row is None:
            return False
        self._db.delete(row)
        self._db.commit()
        return True
