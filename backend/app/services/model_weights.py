"""Delete on-disk weights when a model is removed from the registry."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_IDS = frozenset({"salad", "superpoint", "netvlad", "sift", "lightglue", "sam"})


def _torch_hub_checkpoints_dir() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints"


def _project_models_root() -> Path:
    return settings.project_root / "models"


def weight_paths_for_model(model_id: str, weights_path: Optional[str] = None) -> List[Path]:
    """Resolve concrete weight file paths that may be deleted for a model."""
    paths: List[Path] = []

    if model_id == "salad":
        paths.append(Path(settings.salad_ckpt_path))
    elif model_id == "netvlad":
        paths.append(Path(settings.netvlad_ckpt_path))
    elif model_id == "sam":
        paths.append(Path(settings.sam_ckpt_path))
    elif model_id == "superpoint":
        hub_dir = _torch_hub_checkpoints_dir()
        paths.extend(
            [
                hub_dir / "superpoint_v1.pth",
                hub_dir / "superpoint_lightglue.pth",
            ]
        )
    elif model_id == "lightglue":
        hub_dir = _torch_hub_checkpoints_dir()
        paths.extend(
            [
                hub_dir / "superpoint_lightglue_v0-1_arxiv.pth",
                hub_dir / "superpoint_lightglue.pth",
            ]
        )
    elif model_id == "sift":
        return []

    if weights_path:
        candidate = Path(weights_path)
        if candidate.is_file() and candidate not in paths:
            paths.append(candidate)

    return paths


def _is_safe_to_delete(path: Path) -> bool:
    resolved = path.resolve()
    allowed_roots = [
        _project_models_root().resolve(),
        _torch_hub_checkpoints_dir().resolve(),
    ]
    return any(resolved == root or root in resolved.parents for root in allowed_roots)


def delete_model_weights(model_id: str, weights_path: Optional[str] = None) -> List[str]:
    """Delete weight files for a model. Returns list of deleted paths (as strings)."""
    deleted: List[str] = []
    models_root = _project_models_root()

    for path in weight_paths_for_model(model_id, weights_path):
        if not path.exists():
            continue
        if not _is_safe_to_delete(path):
            logger.warning("Skip unsafe weight path for %s: %s", model_id, path)
            continue
        try:
            path.unlink()
            deleted.append(str(path))
            logger.info("Deleted weight file: %s", path)
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", path, exc)

    model_dir = models_root / model_id
    if model_dir.is_dir() and _is_safe_to_delete(model_dir):
        try:
            shutil.rmtree(model_dir)
            deleted.append(str(model_dir))
            logger.info("Deleted model directory: %s", model_dir)
        except OSError as exc:
            logger.warning("Failed to delete directory %s: %s", model_dir, exc)

    return deleted
