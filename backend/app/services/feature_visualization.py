"""Feature visualization: keypoint extraction for heatmap overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from app.services.sift.engine import SIFTEngine
from app.services.superpoint.engine import SuperPointEngine

KEYPOINT_MODELS = frozenset({"superpoint", "sift"})


def _percentile(values: np.ndarray, p: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, p))


def get_auto_crop_bbox(rgb: np.ndarray, threshold: float = 15.0) -> Tuple[int, int, int, int]:
    """Return (y1, y2, x1, x2) inner visible region, skipping black borders."""
    gray = rgb.mean(axis=2)
    mask = gray > threshold
    h, w = mask.shape
    if not mask.any():
        return 0, h, 0, w

    center_y, center_x = h // 2, w // 2
    row = mask[center_y, :]
    col = mask[:, center_x]

    if row.any() and col.any():
        xs = np.where(row)[0]
        ys = np.where(col)[0]
        inner_min_x, inner_max_x = int(xs[0]), int(xs[-1])
        inner_min_y, inner_max_y = int(ys[0]), int(ys[-1])
    else:
        ys, xs = np.where(mask)
        inner_min_x, inner_max_x = int(xs.min()), int(xs.max())
        inner_min_y, inner_max_y = int(ys.min()), int(ys.max())

    pad = 5
    y1 = min(inner_min_y + pad, center_y)
    y2 = max(inner_max_y - pad, center_y)
    x1 = min(inner_min_x + pad, center_x)
    x2 = max(inner_max_x - pad, center_x)
    return y1, y2, x1, x2


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def run_feature_visualization(image_path: str | Path, model_id: str) -> Dict[str, Any]:
    path = Path(image_path)
    if model_id not in KEYPOINT_MODELS:
        raise ValueError(
            "Model '{}' does not produce keypoints. Use superpoint or sift.".format(model_id)
        )

    rgb = _load_rgb(path)
    height, width = rgb.shape[:2]
    crop_bbox = list(get_auto_crop_bbox(rgb))
    y1, y2, x1, x2 = crop_bbox

    if model_id == "superpoint":
        feats = SuperPointEngine.get_shared().extract_image_for_visualization(path)
        keypoints = feats["keypoints"]
        scores = feats["scores"].astype(np.float64)
        backend = "superpoint"
    elif model_id == "sift":
        feats = SIFTEngine.get_shared().extract_image_for_visualization(path)
        keypoints = feats["keypoints"]
        scores = feats["scores"].astype(np.float64)
        backend = "sift"
    else:
        raise ValueError("Unsupported keypoint model '{}'".format(model_id))

    if keypoints.size == 0:
        raise RuntimeError("No keypoints detected in image")

    edge_margin = 5
    valid = (
        (keypoints[:, 0] >= x1 + edge_margin)
        & (keypoints[:, 0] <= x2 - edge_margin)
        & (keypoints[:, 1] >= y1 + edge_margin)
        & (keypoints[:, 1] <= y2 - edge_margin)
    )
    keypoints = keypoints[valid]
    scores = scores[valid]

    if keypoints.size == 0:
        raise RuntimeError("No keypoints inside valid image region")

    kp_list: List[List[float]] = keypoints.tolist()
    score_list: List[float] = scores.tolist()

    return {
        "image_width": width,
        "image_height": height,
        "keypoints": kp_list,
        "scores": score_list,
        "crop_bbox": crop_bbox,
        "keypoint_count": len(score_list),
        "confidence_max": float(scores.max()),
        "confidence_min": float(scores.min()),
        "confidence_median": float(_percentile(scores, 50)),
        "model_id": model_id,
        "backend": backend,
        "message": (
            "SuperPoint keypoints"
            if backend == "superpoint"
            else "SIFT keypoints"
        ),
    }
