"""Persist / load SuperPoint local features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

MANIFEST_NAME = "manifest.json"
FEATURES_DIR_NAME = "keypoints"


def save_feature_index(
    output_dir: Path,
    *,
    model_id: str,
    image_records: List[Dict[str, str]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": model_id,
        "feature_type": "local_superpoint",
        "count": len(image_records),
        "images": image_records,
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def save_image_features(features_dir: Path, image_id: str, data: Dict[str, np.ndarray]) -> Path:
    features_dir.mkdir(parents=True, exist_ok=True)
    out_path = features_dir / "{}.npz".format(image_id)
    np.savez_compressed(out_path, **data)
    return out_path


def load_manifest(feature_dir: Path) -> Dict[str, Any]:
    manifest_path = Path(feature_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError("SuperPoint manifest not found: {}".format(manifest_path))
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def feature_index_exists(feature_dir: Path) -> bool:
    feature_dir = Path(feature_dir)
    return (feature_dir / MANIFEST_NAME).is_file()


def features_dir_for(feature_dir: Path) -> Path:
    return Path(feature_dir) / FEATURES_DIR_NAME
