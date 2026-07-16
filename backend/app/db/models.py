import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.compat import UTC
from app.db.base import Base
from app.db.enums import SceneStatus


class Scene(Base):
    """Persistent scene record with assets paths and index load status."""

    __tablename__ = "scenes"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    dataset_family: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    scene_slug: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    point_cloud_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    feature_index_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    reference_images_dir: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    train_image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[SceneStatus] = mapped_column(
        Enum(SceneStatus, native_enum=False, length=32),
        nullable=False,
        default=SceneStatus.unloaded,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
