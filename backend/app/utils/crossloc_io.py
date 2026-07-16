"""Parse CrossLoc / ACE dataset calibration and pose files."""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Union

import numpy as np

PoseMatrix = Union[list, np.ndarray]


def parse_calibration_text(text: str) -> Dict[str, Optional[float]]:
    """
    Parse ACE/CrossLoc calibration file content.

    Supports:
    - Single float: focal length in pixels (original image resolution)
    - 3x3 matrix: full camera intrinsics K
    """
    data = np.loadtxt(io.StringIO(text.strip()))
    if data.ndim == 0 or data.size == 1:
        return {
            "focal_length": float(data),
            "pp_x": None,
            "pp_y": None,
        }
    if data.shape == (3, 3):
        return {
            "focal_length": float(data[0, 0]),
            "focal_length_y": float(data[1, 1]),
            "pp_x": float(data[0, 2]),
            "pp_y": float(data[1, 2]),
        }
    raise ValueError(
        "Calibration file must contain a single focal length or a 3x3 intrinsics matrix"
    )


def parse_calibration_file(path: str) -> Dict[str, Optional[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_calibration_text(f.read())


def scale_intrinsics_for_resize(
    calibration: Dict[str, Optional[float]],
    orig_height: float,
    resized_width: float,
    resized_height: float,
) -> Dict[str, float]:
    """
    Scale calibration from original image space to resized image (ACE CamLocDataset logic).
    """
    scale = resized_height / orig_height
    focal = calibration["focal_length"] * scale
    if calibration.get("pp_x") is not None and calibration.get("pp_y") is not None:
        pp_x = calibration["pp_x"] * scale
        pp_y = calibration["pp_y"] * scale
    else:
        pp_x = resized_width / 2.0
        pp_y = resized_height / 2.0

    return {
        "focal_length": focal,
        "pp_x": pp_x,
        "pp_y": pp_y,
        "image_width": float(resized_width),
        "image_height": float(resized_height),
        "focal_length_original": float(calibration["focal_length"]),
        "scale_factor": float(scale),
    }


def parse_pose_text(text: str) -> np.ndarray:
    """Parse 4x4 pose matrix from CrossLoc poses/*.txt file."""
    pose = np.loadtxt(io.StringIO(text.strip()))
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError("Pose file must be a 4x4 matrix (or 16 values)")
    return pose.astype(np.float64)


def parse_pose_file(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        return parse_pose_text(f.read())


def pose_matrix_to_list(pose: np.ndarray) -> list:
    return pose.tolist()
