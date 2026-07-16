"""ACE-based single-image localization service."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.cuda.amp import autocast

from app.services.scene_constants import DEFAULT_SCENE_ID, SCENE_HEAD_WEIGHTS

logger = logging.getLogger(__name__)

_cv2 = None


def _get_cv2():
    """Lazy import OpenCV so the API can start when cv2/libpng is misconfigured."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2 as cv2_module
        except (ImportError, OSError) as exc:
            raise ImportError(
                "OpenCV (cv2) is required for ACE scene coordinate regression (PnP). "
                "In the sup conda env run: bash scripts/fix_sup_opencv.sh"
            ) from exc
        _cv2 = cv2_module
    return _cv2

try:
    import dsacstar as _dsacstar

    DSACSTAR_AVAILABLE = True
except ImportError:
    _dsacstar = None
    DSACSTAR_AVAILABLE = False
    logger.warning(
        "dsacstar not installed; pose estimation will use OpenCV solvePnPRansac fallback. "
        "Install with: cd ace-main/dsacstar && CONDA_PREFIX=$CONDA_PREFIX pip install ."
    )

# ---------------------------------------------------------------------------
# ACE codebase path (ace-main lives alongside scene-understanding-platform)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ACE_ROOT = _PROJECT_ROOT.parent / "ace-main"
if str(_ACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACE_ROOT))

from ace_network import Regressor  # noqa: E402
from app.utils.crossloc_io import scale_intrinsics_for_resize  # noqa: E402

# DSAC* RANSAC defaults (same as test_ace.py)
DEFAULT_RANSAC_HYPOTHESES = 64
DEFAULT_INLIER_THRESHOLD = 10.0
DEFAULT_INLIER_ALPHA = 100.0
DEFAULT_MAX_PIXEL_ERROR = 100.0
DEFAULT_IMAGE_HEIGHT = 480


class LocalizationService:
    """Single-image localization using ACE scene coordinate regression + DSAC* PnP."""

    def __init__(
        self,
        models_dir: Path | None = None,
        encoder_path: Path | None = None,
        device: str = "cuda:0",
        image_height: int = DEFAULT_IMAGE_HEIGHT,
    ) -> None:
        self.models_dir = models_dir or (_PROJECT_ROOT / "models" / "ace")
        self.encoder_path = encoder_path or (
            _PROJECT_ROOT / "models" / "weights" / "ace_encoder_pretrained.pt"
        )
        self.device = torch.device(device)
        self.image_height = image_height

        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU inference")
            self.device = torch.device("cpu")

        self._encoder_state: dict | None = None
        self._networks: dict[str, Regressor] = {}

        self._load_encoder()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"ACE encoder weights not found: {self.encoder_path}")
        self._encoder_state = torch.load(self.encoder_path, map_location="cpu")
        logger.info("Loaded ACE encoder from %s", self.encoder_path)

    def _resolve_head_path(self, scene_id: str) -> Path:
        filename = SCENE_HEAD_WEIGHTS.get(scene_id)
        if filename is None:
            available = ", ".join(sorted(SCENE_HEAD_WEIGHTS))
            raise ValueError(
                f"Unknown scene_id '{scene_id}'. Available: {available}"
            )
        head_path = self.models_dir / filename
        if not head_path.exists():
            raise FileNotFoundError(f"ACE head weights not found: {head_path}")
        return head_path

    def _get_network(self, scene_id: str) -> Regressor:
        if scene_id in self._networks:
            return self._networks[scene_id]

        head_path = self._resolve_head_path(scene_id)
        head_state = torch.load(head_path, map_location="cpu")
        network = Regressor.create_from_split_state_dict(self._encoder_state, head_state)
        network = network.to(self.device)
        network.eval()
        self._networks[scene_id] = network
        logger.info("Loaded ACE head for scene '%s' from %s", scene_id, head_path)
        return network

    def preload_scene(self, scene_id: str = DEFAULT_SCENE_ID) -> None:
        """Eagerly load a scene head onto GPU."""
        self._get_network(scene_id)

    @staticmethod
    def list_scenes() -> list[dict[str, str]]:
        return [
            {"scene_id": sid, "weights_file": fname}
            for sid, fname in SCENE_HEAD_WEIGHTS.items()
        ]

    # ------------------------------------------------------------------
    # Image preprocessing (mirrors CamLocDataset, mode=0, no augment)
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_image(image: np.ndarray, image_height: int) -> np.ndarray:
        scale = image_height / image.shape[0]
        new_w = int(round(image.shape[1] * scale))
        pil = Image.fromarray(image)
        resized = pil.resize((new_w, image_height), Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)

    @staticmethod
    def _to_model_tensor(image_rgb: np.ndarray) -> torch.Tensor:
        """Grayscale + normalize to ACE training statistics."""
        gray = (
            0.299 * image_rgb[:, :, 0]
            + 0.587 * image_rgb[:, :, 1]
            + 0.114 * image_rgb[:, :, 2]
        ).astype(np.float32) / 255.0
        gray = (gray - 0.4) / 0.25
        # Use torch.tensor(list) to avoid torch.from_numpy when NumPy ABI mismatches PyTorch.
        return torch.tensor(gray.tolist(), dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _load_rgb_image(image_path: Path) -> np.ndarray:
        """Load RGB uint8 image via PIL (avoids cv2 imgcodecs / libjxl at read time)."""
        with Image.open(image_path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def preprocess_image(self, image_path: str | Path) -> tuple[torch.Tensor, dict[str, float], float]:
        """
        Load and preprocess a single RGB image.

        Returns:
            image_1HW: (1, 1, H, W) tensor ready for the network
            intrinsics: dict with focal_length, pp_x, pp_y, image_width, image_height
            orig_height: original image height before resize (for calibration scaling)
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = self._load_rgb_image(image_path)

        orig_h, orig_w = image.shape[:2]
        scale = self.image_height / orig_h
        image = self._resize_image(image, self.image_height)
        tensor = self._to_model_tensor(image).unsqueeze(0)  # (1, 1, H, W)

        resized_h, resized_w = tensor.shape[2], tensor.shape[3]
        focal_length = orig_w * scale  # approximate; override via API if known
        pp_x = resized_w / 2.0
        pp_y = resized_h / 2.0

        intrinsics = {
            "focal_length": focal_length,
            "pp_x": pp_x,
            "pp_y": pp_y,
            "image_width": float(resized_w),
            "image_height": float(resized_h),
        }
        return tensor, intrinsics, float(orig_h)

    # ------------------------------------------------------------------
    # Inference + pose estimation (follows test_ace.py)
    # ------------------------------------------------------------------

    def _forward(self, network: Regressor, image_1HW: torch.Tensor) -> torch.Tensor:
        image_1HW = image_1HW.to(self.device, non_blocking=True)
        with torch.no_grad(), autocast(enabled=self.device.type == "cuda"):
            scene_coords = network(image_1HW)
        return scene_coords.float().cpu()

    def _estimate_pose(
        self,
        scene_coordinates_3HW: torch.Tensor,
        network: Regressor,
        intrinsics: dict[str, float],
        ransac_hypotheses: int = DEFAULT_RANSAC_HYPOTHESES,
        inlier_threshold: float = DEFAULT_INLIER_THRESHOLD,
        inlier_alpha: float = DEFAULT_INLIER_ALPHA,
        max_pixel_error: float = DEFAULT_MAX_PIXEL_ERROR,
    ) -> tuple[torch.Tensor, int]:
        if DSACSTAR_AVAILABLE:
            return self._estimate_pose_dsacstar(
                scene_coordinates_3HW,
                network,
                intrinsics,
                ransac_hypotheses,
                inlier_threshold,
                inlier_alpha,
                max_pixel_error,
            )
        return self._estimate_pose_opencv(
            scene_coordinates_3HW,
            network.OUTPUT_SUBSAMPLE,
            intrinsics,
        )

    def _estimate_pose_dsacstar(
        self,
        scene_coordinates_3HW: torch.Tensor,
        network: Regressor,
        intrinsics: dict[str, float],
        ransac_hypotheses: int,
        inlier_threshold: float,
        inlier_alpha: float,
        max_pixel_error: float,
    ) -> tuple[torch.Tensor, int]:
        out_pose = torch.zeros((4, 4))
        inlier_count = _dsacstar.forward_rgb(
            scene_coordinates_3HW.unsqueeze(0),
            out_pose,
            ransac_hypotheses,
            inlier_threshold,
            intrinsics["focal_length"],
            intrinsics["pp_x"],
            intrinsics["pp_y"],
            inlier_alpha,
            max_pixel_error,
            network.OUTPUT_SUBSAMPLE,
        )
        return out_pose, int(inlier_count)

    @staticmethod
    def _estimate_pose_opencv(
        scene_coordinates_3HW: torch.Tensor,
        output_subsample: int,
        intrinsics: dict[str, float],
    ) -> tuple[torch.Tensor, int]:
        """Fallback PnP when dsacstar is unavailable (matches ACE 2D-3D sampling grid)."""
        sc = scene_coordinates_3HW.numpy()
        _, h, w = sc.shape
        obj_pts: list[np.ndarray] = []
        img_pts: list[list[float]] = []

        offset = output_subsample // 2
        for y in range(h):
            for x in range(w):
                pt3 = sc[:, y, x]
                if np.linalg.norm(pt3) < 1e-3:
                    continue
                obj_pts.append(pt3)
                img_pts.append([
                    x * output_subsample + offset,
                    y * output_subsample + offset,
                ])

        if len(obj_pts) < 6:
            raise RuntimeError("Not enough valid scene coordinate points for PnP")

        obj_np = np.asarray(obj_pts, dtype=np.float64)
        img_np = np.asarray(img_pts, dtype=np.float64)
        k = np.array([
            [intrinsics["focal_length"], 0.0, intrinsics["pp_x"]],
            [0.0, intrinsics["focal_length"], intrinsics["pp_y"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        cv2 = _get_cv2()
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_np,
            img_np,
            k,
            None,
            iterationsCount=256,
            reprojectionError=10.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            raise RuntimeError("OpenCV solvePnPRansac failed")

        rot = Rotation.from_rotvec(rvec.reshape(3)).as_matrix()
        pose_w2c = np.eye(4, dtype=np.float64)
        pose_w2c[:3, :3] = rot
        pose_w2c[:3, 3] = tvec.flatten()
        # ACE stores camera-to-world; OpenCV returns world-to-camera.
        pose_c2w = np.linalg.inv(pose_w2c)

        inlier_count = int(len(inliers)) if inliers is not None else 0
        return torch.tensor(pose_c2w.astype(np.float32).tolist()), inlier_count

    @staticmethod
    def _build_pose_result(out_pose: torch.Tensor, inlier_count: int) -> dict[str, Any]:
        """Convert 4x4 pose tensor to a result dict with R, t, and confidence."""
        pose_np = out_pose.numpy()
        rotation_matrix = pose_np[:3, :3]
        translation = pose_np[:3, 3]

        rotation_vector = Rotation.from_matrix(rotation_matrix).as_rotvec()

        return {
            "pose_matrix": pose_np.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "translation": translation.tolist(),
            "rotation_vector": rotation_vector.tolist(),
            "inlier_count": inlier_count,
            "confidence": float(inlier_count),
        }

    @staticmethod
    def _compute_pose_errors(
        gt_pose: np.ndarray,
        est_pose: np.ndarray,
    ) -> dict[str, float]:
        t_err = float(np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3]))
        gt_r = gt_pose[:3, :3]
        est_r = est_pose[:3, :3]
        r_delta = est_r @ gt_r.T
        r_vec = Rotation.from_matrix(r_delta).as_rotvec()
        r_err = float(np.linalg.norm(r_vec) * 180.0 / math.pi)
        return {"rotation_error_deg": r_err, "translation_error_m": t_err}

    @staticmethod
    def parse_gt_pose(raw: list[float] | list[list[float]] | None) -> np.ndarray | None:
        """Parse optional ground-truth pose (4x4 flat list or 4x4 nested list)."""
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape == (16,):
            return arr.reshape(4, 4)
        if arr.shape == (4, 4):
            return arr
        raise ValueError("gt_pose must be a 4x4 matrix or flat list of 16 floats")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(
        self,
        image_path: str | Path,
        scene_id: str = DEFAULT_SCENE_ID,
        focal_length: float | None = None,
        pp_x: float | None = None,
        pp_y: float | None = None,
        gt_pose: list[float] | list[list[float]] | None = None,
        calibration: dict | None = None,
    ) -> dict[str, Any]:
        """
        Run full localization pipeline on a single image.

        Args:
            image_path: Path to query image.
            scene_id: Scene identifier (maps to a head .pt file).
            focal_length: Optional manual focal length (original image pixels).
            pp_x, pp_y: Optional principal point (original image pixels).
            gt_pose: Optional 4x4 ground-truth pose for error evaluation.
            calibration: Optional parsed calibration dict from CrossLoc file
                         (focal_length, optional pp_x/pp_y in original image space).

        Returns:
            Dict with pose_matrix, rotation_matrix, translation, rotation_vector,
            inlier_count, confidence, intrinsics, scene_id, and optional gt_errors.
        """
        network = self._get_network(scene_id)
        image_tensor, intrinsics, orig_h = self.preprocess_image(image_path)

        cal_input = calibration
        if cal_input is None and focal_length is not None:
            cal_input = {"focal_length": focal_length, "pp_x": pp_x, "pp_y": pp_y}

        if cal_input is not None:
            scaled = scale_intrinsics_for_resize(
                cal_input,
                orig_height=orig_h,
                resized_width=intrinsics["image_width"],
                resized_height=intrinsics["image_height"],
            )
            intrinsics.update(scaled)

        scene_coords_b3hw = self._forward(network, image_tensor)
        scene_coords_3hw = scene_coords_b3hw[0]

        out_pose, inlier_count = self._estimate_pose(
            scene_coords_3hw, network, intrinsics
        )

        result = self._build_pose_result(out_pose, inlier_count)
        result["scene_id"] = scene_id
        result["image_path"] = str(image_path)
        result["intrinsics"] = intrinsics
        result["pose_backend"] = "dsacstar" if DSACSTAR_AVAILABLE else "opencv"

        gt = self.parse_gt_pose(gt_pose)
        if gt is not None:
            result["gt_errors"] = self._compute_pose_errors(gt, out_pose.numpy())

        return result
