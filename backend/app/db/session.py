from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sqlalchemy import text

from app.config import settings
from app.db.base import Base

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

_SCENE_EXTRA_COLUMNS = (
    ("dataset_family", "VARCHAR(32)"),
    ("scene_slug", "VARCHAR(64)"),
)


def _migrate_scene_columns() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(scenes)")).fetchall()
        existing = {row[1] for row in rows}
        for column, col_type in _SCENE_EXTRA_COLUMNS:
            if column not in existing:
                conn.execute(text(f"ALTER TABLE scenes ADD COLUMN {column} {col_type}"))
        conn.commit()


def init_db() -> None:
    """Create database tables (dev / first-run helper)."""
    # Import models so metadata is populated before create_all.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_scene_columns()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
