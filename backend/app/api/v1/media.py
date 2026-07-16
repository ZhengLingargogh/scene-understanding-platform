"""Serve local image files for retrieval previews (restricted roots)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter()


def _is_allowed_path(path: Path) -> bool:
    resolved = path.resolve()
    for root in settings.media_allowed_roots:
        root_path = Path(root).resolve()
        try:
            resolved.relative_to(root_path)
            return True
        except ValueError:
            continue
    return False


@router.get("/file")
async def serve_local_file(path: str = Query(..., description="Absolute or project-relative file path")):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (settings.project_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    if not _is_allowed_path(candidate):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Path not allowed")

    return FileResponse(candidate)
