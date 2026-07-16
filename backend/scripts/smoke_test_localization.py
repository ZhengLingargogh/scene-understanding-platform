#!/usr/bin/env python3
"""Smoke test for ACE LocalizationService."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.localization import LocalizationService  # noqa: E402

IMAGE = Path(
    "/media/zl/A882F45482F42888/datasets/crossloc/nature/test/rgb/"
    "naturescape-2020-11-06-piloted_00000_DJI_0003.png"
)
GT_POSE_FILE = Path(
    "/media/zl/A882F45482F42888/datasets/crossloc/nature/test/poses/"
    "naturescape-2020-11-06-piloted_00000_DJI_0003.txt"
)
CALIB = 480.0


def main() -> None:
    if not IMAGE.exists():
        raise SystemExit(f"Test image not found: {IMAGE}")

    gt_pose = np.loadtxt(GT_POSE_FILE).tolist()
    service = LocalizationService()
    result = service.localize(
        image_path=IMAGE,
        scene_id="nature",
        focal_length=CALIB,
        gt_pose=gt_pose,
    )
    print(json.dumps({k: result[k] for k in (
        "scene_id", "pose_backend", "inlier_count", "confidence", "gt_errors",
        "translation", "rotation_vector",
    )}, indent=2))


if __name__ == "__main__":
    main()
