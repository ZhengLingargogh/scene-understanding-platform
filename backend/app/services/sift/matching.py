"""SIFT descriptor matching (OpenCV BFMatcher + Lowe ratio test)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.services.detectors.base import KeypointDetectionResult

LOWE_RATIO = 0.75


def _scale_keypoints_to_original(
    keypoints: np.ndarray,
    processed_shape: Tuple[int, int],
    original_size: Tuple[int, int],
) -> List[List[float]]:
    proc_w, proc_h = processed_shape
    orig_w, orig_h = original_size
    if proc_w <= 0 or proc_h <= 0 or keypoints.size == 0:
        return keypoints.tolist()
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h
    scaled = keypoints.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled.tolist()


def _descriptors_from_detection(detection: KeypointDetectionResult) -> np.ndarray:
    desc = detection.feats["descriptors"]
    arr = desc.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] == detection.feats["keypoints"].shape[1]:
        return arr.astype(np.float32)
    return arr.transpose(1, 0).astype(np.float32)


def _keypoints_from_detection(detection: KeypointDetectionResult) -> np.ndarray:
    kpts = detection.feats["keypoints"][0].detach().cpu().numpy()
    return kpts.astype(np.float64)


def match_sift_detections(
    detection0: KeypointDetectionResult,
    detection1: KeypointDetectionResult,
) -> Dict[str, object]:
    kpts0 = _keypoints_from_detection(detection0)
    kpts1 = _keypoints_from_detection(detection1)
    desc0 = _descriptors_from_detection(detection0)
    desc1 = _descriptors_from_detection(detection1)

    orig0 = detection0.original_size
    orig1 = detection1.original_size
    proc0 = detection0.processed_shape
    proc1 = detection1.processed_shape

    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        return {
            "match_count": 0,
            "mean_score": 0.0,
            "keypoints0": _scale_keypoints_to_original(kpts0, (proc0[1], proc0[0]), orig0),
            "keypoints1": _scale_keypoints_to_original(kpts1, (proc1[1], proc1[0]), orig1),
            "matches": [],
            "image0_size": [orig0[0], orig0[1]],
            "image1_size": [orig1[0], orig1[1]],
        }

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc0, desc1, k=2)
    good: List[List[int]] = []
    distances: List[float] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < LOWE_RATIO * n.distance:
            good.append([int(m.queryIdx), int(m.trainIdx)])
            distances.append(float(m.distance))

    mean_score = float(np.mean(distances)) if distances else 0.0
    if distances:
        max_d = max(distances)
        mean_score = 1.0 - (mean_score / max(max_d, 1e-6))

    return {
        "match_count": len(good),
        "mean_score": mean_score,
        "keypoints0": _scale_keypoints_to_original(kpts0, (proc0[1], proc0[0]), orig0),
        "keypoints1": _scale_keypoints_to_original(kpts1, (proc1[1], proc1[0]), orig1),
        "matches": good,
        "image0_size": [orig0[0], orig0[1]],
        "image1_size": [orig1[0], orig1[1]],
    }
