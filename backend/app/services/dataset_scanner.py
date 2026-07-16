"""Scan filesystem dataset roots into catalog structures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from app.services.dataset_catalog import BUILTIN_FAMILY_IDS, SPLITS, DatasetFamily, SceneInfo, SplitInfo
from app.services.salad.image_utils import IMAGE_EXTENSIONS

_CALIBRATION_DIR = "calibration"
_POSES_DIR = "poses"
_RGB_DIR = "rgb"
_PROJECT_FEATURES = "data/features"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    normalized = value.strip().lower()
    slug = _SLUG_PATTERN.sub("_", normalized).strip("_")
    return slug or "dataset"


def count_rgb_images(rgb_dir: Path) -> int:
    if not rgb_dir.is_dir():
        return 0
    return sum(
        1
        for path in rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_split(split_root: Path) -> Tuple[bool, str, int]:
    calibration_dir = split_root / _CALIBRATION_DIR
    poses_dir = split_root / _POSES_DIR
    rgb_dir = split_root / _RGB_DIR

    if not calibration_dir.is_dir():
        return False, f"缺少 {_CALIBRATION_DIR} 目录", 0
    if not poses_dir.is_dir():
        return False, f"缺少 {_POSES_DIR} 目录", 0
    if not rgb_dir.is_dir():
        return False, f"缺少 {_RGB_DIR} 目录", 0

    image_count = count_rgb_images(rgb_dir)
    if image_count == 0:
        return False, f"{_RGB_DIR} 目录下无图像", 0

    return True, "", image_count


def _build_split_info(
    family_id: str,
    scene_id: str,
    split: str,
    split_root: Path,
    image_count: int,
) -> SplitInfo:
    dataset_id = f"{family_id}-{scene_id}-{split}"
    rgb_dir = split_root / _RGB_DIR
    return {
        "split": split,
        "label": "训练集" if split == "train" else "测试集",
        "rgb_dir": str(rgb_dir.resolve()),
        "calibration_dir": str((split_root / _CALIBRATION_DIR).resolve()),
        "poses_dir": str((split_root / _POSES_DIR).resolve()),
        "image_count": image_count,
        "dataset_id": dataset_id,
        "default_feature_dir": f"{_PROJECT_FEATURES}/{dataset_id}",
    }


def _scan_scene(
    family_id: str,
    scene_path: Path,
) -> Tuple[SceneInfo | None, List[str]]:
    warnings: List[str] = []
    scene_name = scene_path.name
    scene_id = slugify(scene_name)

    if not scene_path.is_dir():
        return None, [f"{scene_name}: 不是目录"]

    splits: List[SplitInfo] = []
    for split in SPLITS:
        split_root = scene_path / split
        if not split_root.is_dir():
            warnings.append(f"{scene_name}/{split}: 目录不存在")
            return None, warnings

        valid, reason, image_count = _validate_split(split_root)
        if not valid:
            warnings.append(f"{scene_name}/{split}: {reason}")
            return None, warnings

        splits.append(_build_split_info(family_id, scene_id, split, split_root, image_count))

    return (
        {
            "scene_id": scene_id,
            "name": scene_name,
            "path": str(scene_path.resolve()),
            "splits": splits,
        },
        warnings,
    )


def scan_dataset_root(root_path: str) -> Tuple[DatasetFamily, List[str], bool]:
    """
    Scan a dataset root directory.

    Returns (family, warnings, valid) where valid means at least one legal scene.
    """
    resolved_root = Path(root_path).expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"路径不存在或不是目录: {root_path}")

    family_id = slugify(resolved_root.name)
    if family_id in BUILTIN_FAMILY_IDS:
        raise ValueError(f'family_id "{family_id}" 与内置数据集冲突，请更换目录名')

    warnings: List[str] = []
    scenes: List[SceneInfo] = []

    for entry in sorted(resolved_root.iterdir(), key=lambda path: path.name.lower()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue

        scene_info, scene_warnings = _scan_scene(family_id, entry)
        warnings.extend(scene_warnings)
        if scene_info is not None:
            scenes.append(scene_info)

    if not scenes:
        warnings.append("未找到合法场景（每个场景需含 train/test，且各有 calibration、poses、rgb 图像）")

    family: DatasetFamily = {
        "id": family_id,
        "name": resolved_root.name,
        "root_path": str(resolved_root),
        "scenes": scenes,
    }
    return family, warnings, len(scenes) > 0
