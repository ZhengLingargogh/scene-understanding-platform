"""Image discovery and preprocessing helpers for SALAD."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(directory: Path) -> List[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError("Image directory not found: {}".format(root))

    images = [
        path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError("No images found in {}".format(root))
    return images


def build_image_records(image_paths: Sequence[Path]) -> List[dict]:
    records = []
    for index, path in enumerate(image_paths):
        resolved = path.resolve()
        records.append(
            {
                "id": str(index),
                "path": str(resolved),
                "filename": path.name,
            }
        )
    return records
