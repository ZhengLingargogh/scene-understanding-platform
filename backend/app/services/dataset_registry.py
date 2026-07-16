"""Persist and load custom dataset families."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from app.config import settings
from app.services.dataset_catalog import DatasetFamily, invalidate_family_cache
from app.services.dataset_scanner import scan_dataset_root
from app.services.mock_datasets import rebuild_mock_datasets_index

_STORE_VERSION = 1


def _store_path() -> Path:
    return settings.data_dir / "custom_datasets.json"


def _read_store() -> dict:
    path = _store_path()
    if not path.is_file():
        return {"version": _STORE_VERSION, "families": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"version": _STORE_VERSION, "families": []}
    data.setdefault("families", [])
    return data


def _write_store(data: dict) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_custom_families() -> List[DatasetFamily]:
    store = _read_store()
    families: List[DatasetFamily] = []
    for item in store.get("families", []):
        if not isinstance(item, dict):
            continue
        scenes = item.get("scenes", [])
        if not scenes:
            continue
        families.append(
            {
                "id": str(item["id"]),
                "name": str(item["name"]),
                "root_path": str(item["root_path"]),
                "scenes": scenes,
            }
        )
    return families


def list_custom_summaries() -> List[dict]:
    store = _read_store()
    summaries: List[dict] = []
    for item in store.get("families", []):
        if not isinstance(item, dict):
            continue
        summaries.append(
            {
                "family_id": str(item["id"]),
                "name": str(item["name"]),
                "root_path": str(item["root_path"]),
                "scene_count": len(item.get("scenes", [])),
                "registered_at": item.get("registered_at", datetime.now(timezone.utc).isoformat()),
            }
        )
    return summaries


def family_exists(family_id: str) -> bool:
    store = _read_store()
    return any(item.get("id") == family_id for item in store.get("families", []))


def register_dataset_from_path(root_path: str) -> tuple[DatasetFamily, List[str], datetime]:
    family, warnings, valid = scan_dataset_root(root_path)
    if not valid:
        detail = warnings[-1] if warnings else "无合法场景"
        raise ValueError(detail)

    store = _read_store()
    families: list = store.setdefault("families", [])
    if any(item.get("id") == family["id"] for item in families):
        raise ValueError(f'数据集 "{family["id"]}" 已注册')
    if any(item.get("root_path") == family["root_path"] for item in families):
        raise ValueError(f'路径已注册: {family["root_path"]}')

    registered_at = datetime.now(timezone.utc)
    families.append(
        {
            "id": family["id"],
            "name": family["name"],
            "root_path": family["root_path"],
            "scenes": family["scenes"],
            "registered_at": registered_at.isoformat(),
        }
    )
    _write_store(store)
    invalidate_family_cache()
    rebuild_mock_datasets_index()
    return family, warnings, registered_at


def delete_custom_family(family_id: str) -> bool:
    store = _read_store()
    families: list = store.get("families", [])
    new_families = [item for item in families if item.get("id") != family_id]
    if len(new_families) == len(families):
        return False
    store["families"] = new_families
    _write_store(store)
    invalidate_family_cache()
    rebuild_mock_datasets_index()
    return True
