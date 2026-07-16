from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.deps import get_scene_service
from app.db.session import get_db
from app.schemas.dataset_registry import (
    BrowseResponse,
    CustomDatasetSummary,
    DatasetScanPreview,
    RegisterDatasetRequest,
    RegisterDatasetResponse,
)
from app.schemas.scene import DatasetCatalogResponse, SceneCreate, SceneResponse, SceneUpdate
from app.services.dataset_catalog import is_builtin_family
from app.services.dataset_catalog_service import build_dataset_catalog
from app.services.dataset_paths import browse_directory
from app.services.dataset_registry import (
    delete_custom_family,
    list_custom_summaries,
    register_dataset_from_path,
)
from app.services.dataset_scanner import scan_dataset_root
from app.services.scene_seed import seed_catalog_scenes
from app.services.scene_service import SceneService

router = APIRouter()


def _family_to_preview(family: dict, warnings: List[str], valid: bool) -> DatasetScanPreview:
    return DatasetScanPreview(
        family_id=family["id"],
        name=family["name"],
        root_path=family["root_path"],
        scenes=family["scenes"],
        warnings=warnings,
        valid=valid,
    )


@router.get("/catalog", response_model=List[DatasetCatalogResponse])
async def list_dataset_catalog(db: Session = Depends(get_db)):
    """Built-in and custom datasets with scene hierarchy and SQLite registration."""
    return build_dataset_catalog(db)


@router.get("/datasets/custom", response_model=List[CustomDatasetSummary])
async def list_custom_datasets():
    return [CustomDatasetSummary(**item) for item in list_custom_summaries()]


@router.get("/datasets/browse", response_model=BrowseResponse)
async def browse_dataset_paths(path: str | None = Query(default=None)):
    try:
        current, parent, entries = browse_directory(path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return BrowseResponse(
        path=str(current),
        parent_path=str(parent) if parent else None,
        entries=entries,
    )


@router.post("/datasets/scan", response_model=DatasetScanPreview)
async def scan_dataset(payload: RegisterDatasetRequest):
    try:
        family, warnings, valid = scan_dataset_root(payload.root_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _family_to_preview(family, warnings, valid)


@router.post("/datasets/register", response_model=RegisterDatasetResponse, status_code=status.HTTP_201_CREATED)
async def register_dataset(
    payload: RegisterDatasetRequest,
    db: Session = Depends(get_db),
):
    try:
        family, warnings, registered_at = register_dataset_from_path(payload.root_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    seed_catalog_scenes(db)
    return RegisterDatasetResponse(
        family_id=family["id"],
        name=family["name"],
        root_path=family["root_path"],
        scene_count=len(family["scenes"]),
        warnings=warnings,
        registered_at=registered_at,
    )


@router.delete("/datasets/{family_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_dataset(
    family_id: str,
    db: Session = Depends(get_db),
    service: SceneService = Depends(get_scene_service),
):
    if is_builtin_family(family_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="内置数据集不可删除")
    if not delete_custom_family(family_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="自定义数据集不存在")
    service.delete_scenes_by_family(family_id)


@router.get("", response_model=List[SceneResponse])
async def list_scenes(service: SceneService = Depends(get_scene_service)):
    return service.list_scenes()


@router.post("", response_model=SceneResponse, status_code=status.HTTP_201_CREATED)
async def create_scene(
    payload: SceneCreate,
    service: SceneService = Depends(get_scene_service),
):
    return service.create_scene(payload)


@router.get("/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: str, service: SceneService = Depends(get_scene_service)):
    scene = service.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
    return scene


@router.patch("/{scene_id}", response_model=SceneResponse)
async def update_scene(
    scene_id: str,
    payload: SceneUpdate,
    service: SceneService = Depends(get_scene_service),
):
    scene = service.update_scene(scene_id, payload)
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
    return scene


@router.delete("/{scene_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scene(scene_id: str, service: SceneService = Depends(get_scene_service)):
    if not service.delete_scene(scene_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
