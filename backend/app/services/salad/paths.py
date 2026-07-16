"""Path resolution helpers shared by SALAD plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from app.config import settings


def resolve_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (settings.project_root / candidate).resolve()


def resolve_feature_dir(
    *,
    scene_feature_index_path: Optional[str] = None,
    dataset_feature_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[Path]:
    for raw in (scene_feature_index_path, output_path, dataset_feature_dir):
        if raw:
            return resolve_path(raw)
    return None
