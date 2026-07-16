"""Dataset catalog for scene management and UI cascades."""

from __future__ import annotations

from typing import Dict, List, TypedDict

from typing_extensions import NotRequired

_CROSSLOC_ROOT = "/media/zl/A882F45482F42888/cursor/work_place/datasets/crossloc"
_UAVD4L_ROOT = "/media/zl/A882F45482F42888/cursor/work_place/datasets/uavd4l"
_PROJECT_FEATURES = "data/features"

CROSSLOC_SCENE_IDS = ("nature", "natureoop", "urban", "urbanoop")

_CROSSLOC_LABELS = {
    "nature": "Nature",
    "natureoop": "Nature OOP",
    "urban": "Urban",
    "urbanoop": "Urban OOP",
}

_UAVD4L_SPECS = (
    ("inTraj", "intraj", "InTraj"),
    ("outTraj", "outtraj", "OutTraj"),
)

_CROSSLOC_COUNTS = {
    "nature": {"train": 845, "test": 1058},
    "natureoop": {"train": 226, "test": 283},
    "urban": {"train": 1263, "test": 1580},
    "urbanoop": {"train": 544, "test": 680},
}

_UAVD4L_COUNTS = {
    "intraj": {"train": 1069, "test": 535},
    "outtraj": {"train": 1461, "test": 731},
}

SPLITS = ("train", "test")


class SplitInfo(TypedDict):
    split: str
    label: str
    rgb_dir: str
    image_count: int
    dataset_id: str
    default_feature_dir: str
    calibration_dir: NotRequired[str]
    poses_dir: NotRequired[str]


class SceneInfo(TypedDict):
    scene_id: str
    name: str
    path: str
    splits: List[SplitInfo]


class DatasetFamily(TypedDict):
    id: str
    name: str
    root_path: str
    scenes: List[SceneInfo]


def _crossloc_rgb(scene: str, split: str) -> str:
    return f"{_CROSSLOC_ROOT}/{scene}/{split}/rgb"


def _uavd4l_rgb(traj_dir: str, split: str) -> str:
    return f"{_UAVD4L_ROOT}/{traj_dir}/{split}/rgb"


def _build_crossloc_family() -> DatasetFamily:
    scenes: List[SceneInfo] = []
    for scene_id in CROSSLOC_SCENE_IDS:
        counts = _CROSSLOC_COUNTS[scene_id]
        splits: List[SplitInfo] = []
        for split in SPLITS:
            dataset_id = f"crossloc-{scene_id}-{split}"
            splits.append(
                {
                    "split": split,
                    "label": "训练集" if split == "train" else "测试集",
                    "rgb_dir": _crossloc_rgb(scene_id, split),
                    "image_count": counts[split],
                    "dataset_id": dataset_id,
                    "default_feature_dir": f"{_PROJECT_FEATURES}/{dataset_id}",
                }
            )
        scenes.append(
            {
                "scene_id": scene_id,
                "name": _CROSSLOC_LABELS[scene_id],
                "path": f"{_CROSSLOC_ROOT}/{scene_id}",
                "splits": splits,
            }
        )
    return {
        "id": "crossloc",
        "name": "CrossLoc",
        "root_path": _CROSSLOC_ROOT,
        "scenes": scenes,
    }


def _build_uavd4l_family() -> DatasetFamily:
    scenes: List[SceneInfo] = []
    for traj_dir, scene_id, label in _UAVD4L_SPECS:
        counts = _UAVD4L_COUNTS[scene_id]
        splits: List[SplitInfo] = []
        for split in SPLITS:
            dataset_id = f"uavd4l-{scene_id}-{split}"
            splits.append(
                {
                    "split": split,
                    "label": "训练集" if split == "train" else "测试集",
                    "rgb_dir": _uavd4l_rgb(traj_dir, split),
                    "image_count": counts[split],
                    "dataset_id": dataset_id,
                    "default_feature_dir": f"{_PROJECT_FEATURES}/{dataset_id}",
                }
            )
        scenes.append(
            {
                "scene_id": scene_id,
                "name": label,
                "path": f"{_UAVD4L_ROOT}/{traj_dir}",
                "splits": splits,
            }
        )
    return {
        "id": "uavd4l",
        "name": "UAVD4L",
        "root_path": _UAVD4L_ROOT,
        "scenes": scenes,
    }


DATASET_FAMILIES: List[DatasetFamily] = [_build_crossloc_family(), _build_uavd4l_family()]

BUILTIN_FAMILY_IDS = frozenset(family["id"] for family in DATASET_FAMILIES)

_FAMILY_BY_ID: Dict[str, DatasetFamily] = {f["id"]: f for f in DATASET_FAMILIES}
_ALL_FAMILIES_CACHE: List[DatasetFamily] | None = None


def invalidate_family_cache() -> None:
    global _ALL_FAMILIES_CACHE
    _ALL_FAMILIES_CACHE = None


def get_all_dataset_families() -> List[DatasetFamily]:
    global _ALL_FAMILIES_CACHE
    if _ALL_FAMILIES_CACHE is not None:
        return list(_ALL_FAMILIES_CACHE)

    from app.services.dataset_registry import load_custom_families

    merged = list(DATASET_FAMILIES) + load_custom_families()
    _ALL_FAMILIES_CACHE = merged
    return list(merged)


def list_dataset_families() -> List[DatasetFamily]:
    return get_all_dataset_families()


def get_dataset_family(family_id: str) -> DatasetFamily | None:
    for family in get_all_dataset_families():
        if family["id"] == family_id:
            return family
    return None


def is_builtin_family(family_id: str) -> bool:
    return family_id in BUILTIN_FAMILY_IDS


def resolve_split_dataset_id(family_id: str, scene_id: str, split: str) -> str | None:
    family = get_dataset_family(family_id)
    if family is None:
        return None
    for scene in family["scenes"]:
        if scene["scene_id"] != scene_id:
            continue
        for item in scene["splits"]:
            if item["split"] == split:
                return item["dataset_id"]
    return None


def resolve_split_rgb_dir(family_id: str, scene_id: str, split: str) -> str | None:
    family = get_dataset_family(family_id)
    if family is None:
        return None
    for scene in family["scenes"]:
        if scene["scene_id"] != scene_id:
            continue
        for item in scene["splits"]:
            if item["split"] == split:
                return item["rgb_dir"]
    return None
