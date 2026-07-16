"""Persist and load SALAD global descriptors for a scene / dataset gallery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MANIFEST_NAME = "manifest.json"
DESCRIPTORS_NAME = "descriptors.npy"


def save_feature_index(
    output_dir: Path,
    *,
    model_id: str,
    image_records: List[Dict[str, str]],
    descriptors: np.ndarray,
    descriptor_dim: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = _l2_normalize(descriptors.astype(np.float32))
    np.save(output_dir / DESCRIPTORS_NAME, normalized)

    manifest = {
        "model_id": model_id,
        "descriptor_dim": descriptor_dim,
        "count": len(image_records),
        "images": image_records,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def load_feature_index(feature_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    feature_dir = Path(feature_dir)
    manifest_path = feature_dir / MANIFEST_NAME
    descriptors_path = feature_dir / DESCRIPTORS_NAME

    if not manifest_path.is_file() or not descriptors_path.is_file():
        raise FileNotFoundError(
            "Feature index not found in {} (expected {} and {})".format(
                feature_dir, MANIFEST_NAME, DESCRIPTORS_NAME
            )
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptors = np.load(descriptors_path)
    return descriptors, manifest


def feature_index_exists(feature_dir: Path) -> bool:
    feature_dir = Path(feature_dir)
    return (feature_dir / MANIFEST_NAME).is_file() and (feature_dir / DESCRIPTORS_NAME).is_file()


def _l2_normalize(descriptors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return descriptors / norms
