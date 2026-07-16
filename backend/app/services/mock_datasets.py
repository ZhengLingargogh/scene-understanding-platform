"""Dataset catalog entries for inference and feature extraction."""

from __future__ import annotations

from typing import Dict, List, Optional

from app.services.dataset_catalog import get_all_dataset_families


def _build_family_datasets(family_id: str) -> List[Dict[str, object]]:
    family = next(family_item for family_item in get_all_dataset_families() if family_item["id"] == family_id)
    entries: List[Dict[str, object]] = []
    prefix = family["name"]
    for scene_info in family["scenes"]:
        scene_id = scene_info["scene_id"]
        label = scene_info["name"]
        for split_info in scene_info["splits"]:
            split = split_info["split"]
            entries.append(
                {
                    "id": split_info["dataset_id"],
                    "name": f"{prefix} {label} {'Train' if split == 'train' else 'Test'}",
                    "family": family_id,
                    "scene_id": scene_id,
                    "split": split,
                    "image_count": split_info["image_count"],
                    "description": f"{prefix} · {label} {'训练集' if split == 'train' else '测试集'}",
                    "reference_rgb_dir": split_info["rgb_dir"] if split == "train" else None,
                    "gallery_rgb_dir": split_info["rgb_dir"],
                    "query_rgb_dir": split_info["rgb_dir"] if split == "test" else None,
                    "default_feature_dir": split_info["default_feature_dir"],
                }
            )
    return entries


def rebuild_mock_datasets_index() -> None:
    global MOCK_DATASETS, _DATASET_BY_ID
    MOCK_DATASETS = []
    for family in get_all_dataset_families():
        MOCK_DATASETS.extend(_build_family_datasets(family["id"]))
    _DATASET_BY_ID = {dataset["id"]: dataset for dataset in MOCK_DATASETS}


MOCK_DATASETS: List[Dict[str, object]] = []
_DATASET_BY_ID: Dict[str, Dict[str, object]] = {}

rebuild_mock_datasets_index()


def list_mock_datasets() -> List[Dict[str, object]]:
    return list(MOCK_DATASETS)


def list_crossloc_datasets() -> List[Dict[str, object]]:
    return [dataset for dataset in MOCK_DATASETS if dataset.get("family") == "crossloc"]


def list_uavd4l_datasets() -> List[Dict[str, object]]:
    return [dataset for dataset in MOCK_DATASETS if dataset.get("family") == "uavd4l"]


def get_mock_dataset(dataset_id: str) -> Optional[Dict[str, object]]:
    return _DATASET_BY_ID.get(dataset_id)


def resolve_input_path(dataset_id: str, input_path: Optional[str] = None) -> str:
    if input_path:
        return input_path
    dataset = get_mock_dataset(dataset_id)
    if dataset is None:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    rgb_dir = (
        dataset.get("reference_rgb_dir")
        or dataset.get("gallery_rgb_dir")
        or dataset.get("query_rgb_dir")
    )
    if not rgb_dir:
        raise ValueError(f"Dataset {dataset_id} has no reference RGB directory for feature extraction")
    return str(rgb_dir)
