from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Scene Understanding Platform"
    app_version: str = "0.1.0"
    debug: bool = True

    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    data_dir: Path = Path("data")
    upload_dir: Path = Path("uploads")
    database_url: str = "sqlite:///./data/scene_platform.db"

    project_root: Path = _PROJECT_ROOT

    # ACE localization
    ace_models_dir: Path = _PROJECT_ROOT / "models" / "ace"
    ace_encoder_path: Path = _PROJECT_ROOT / "models" / "weights" / "ace_encoder_pretrained.pt"
    ace_device: str = "cuda:0"
    ace_image_height: int = 480

    # SALAD (DINOv2 + SALAD aggregator)
    salad_root: Path = _PROJECT_ROOT / "third_party" / "salad"
    salad_ckpt_path: Path = _PROJECT_ROOT / "models" / "salad" / "dino_salad.ckpt"
    salad_device: str = "cuda:0"
    salad_image_size: int = 322
    salad_batch_size: int = 8

    # NetVLAD (VGG16 + NetVLAD)
    netvlad_root: Path = _PROJECT_ROOT / "third_party" / "netvlad"
    netvlad_ckpt_path: Path = _PROJECT_ROOT / "models" / "netvlad" / "vgg16_netvlad.pth"
    netvlad_device: str = "cuda:0"
    netvlad_image_size: tuple = (480, 640)
    netvlad_batch_size: int = 8

    # SuperPoint + LightGlue (third_party/lightglue)
    lightglue_root: Path = _PROJECT_ROOT / "third_party" / "lightglue"
    superpoint_device: str = "cuda:0"
    lightglue_device: str = "cuda:0"
    superpoint_max_keypoints: int = 2048
    superpoint_resize: int = 1024

    # Segment Anything (third_party/segment-anything)
    sam_root: Path = _PROJECT_ROOT / "third_party" / "segment-anything"
    sam_ckpt_path: Path = _PROJECT_ROOT / "models" / "sam" / "sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"
    sam_device: str = "cuda:0"

    media_allowed_roots: List[str] = [
        str(_PROJECT_ROOT),
        str(_PROJECT_ROOT.parent / "datasets"),
        "/media/zl/A882F45482F42888/datasets",
        "/media/zl/A882F45482F42888/cursor/work_place/datasets",
    ]


settings = Settings()
