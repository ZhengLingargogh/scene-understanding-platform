"""Shared helpers for inference API endpoints."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.utils.crossloc_io import (
    parse_calibration_text,
    parse_pose_text,
    pose_matrix_to_list,
)

logger = logging.getLogger(__name__)

_GT_POSE_PLACEHOLDERS = frozenset({"", "string", "null", "none"})


def _has_upload(upload: Optional[UploadFile]) -> bool:
    return upload is not None and bool(upload.filename)


def _is_placeholder_gt_pose(gt_pose: Optional[str]) -> bool:
    if gt_pose is None:
        return True
    return gt_pose.strip().lower() in _GT_POSE_PLACEHOLDERS


def _is_valid_manual_focal(focal_length: Optional[float]) -> bool:
    return focal_length is not None and focal_length > 0


async def save_upload(upload: UploadFile, subdir: str = "") -> Path:
    suffix = Path(upload.filename or "file").suffix or ".bin"
    dest_dir = settings.upload_dir / subdir if subdir else settings.upload_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "{}{}".format(uuid4(), suffix)
    dest.write_bytes(await upload.read())
    return dest


async def resolve_inference_inputs(
    image: UploadFile,
    calibration_file: Optional[UploadFile] = None,
    gt_pose_file: Optional[UploadFile] = None,
    focal_length: Optional[float] = None,
    pp_x: Optional[float] = None,
    pp_y: Optional[float] = None,
    gt_pose: Optional[str] = None,
) -> Tuple[str, Optional[Dict], Optional[List], Dict[str, str]]:
    """Save uploads and resolve intrinsics / GT pose for inference."""
    image_path = str(await save_upload(image, "images"))
    metadata = {"image_path": image_path}

    calibration = None
    if _has_upload(calibration_file):
        cal_bytes = await calibration_file.read()
        try:
            calibration = parse_calibration_text(cal_bytes.decode("utf-8"))
            metadata["calibration_file"] = calibration_file.filename
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid calibration file: {}".format(exc),
            ) from exc
    elif _is_valid_manual_focal(focal_length):
        calibration = {
            "focal_length": focal_length,
            "pp_x": pp_x if pp_x and pp_x > 0 else None,
            "pp_y": pp_y if pp_y and pp_y > 0 else None,
        }
        metadata["focal_length_manual"] = str(focal_length)

    parsed_gt = None
    if _has_upload(gt_pose_file):
        pose_bytes = await gt_pose_file.read()
        try:
            pose_np = parse_pose_text(pose_bytes.decode("utf-8"))
            parsed_gt = pose_matrix_to_list(pose_np)
            metadata["gt_pose_file"] = gt_pose_file.filename
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid pose file: {}".format(exc),
            ) from exc
    elif not _is_placeholder_gt_pose(gt_pose):
        try:
            parsed_gt = json.loads(gt_pose)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid gt_pose JSON: {}".format(exc),
            ) from exc

    return image_path, calibration, parsed_gt, metadata
