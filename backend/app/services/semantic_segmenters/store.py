"""Persist batch semantic segmentation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def save_segmentation_manifest(
    output_dir: Path,
    *,
    model_id: str,
    dataset_id: str,
    image_records: List[Dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    payload = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "image_count": len(image_records),
        "images": image_records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
