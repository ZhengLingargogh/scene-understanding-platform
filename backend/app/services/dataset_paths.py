"""Filesystem helpers for dataset path browsing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from app.config import settings


def _allowed_roots() -> List[Path]:
    roots: List[Path] = []
    for root in settings.media_allowed_roots:
        resolved = Path(root).expanduser().resolve()
        if resolved.is_dir():
            roots.append(resolved)
    return roots


def resolve_allowed_path(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    for root in _allowed_roots():
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    allowed = ", ".join(str(root) for root in _allowed_roots())
    raise ValueError(f"路径不在允许范围内，仅可访问: {allowed}")


def browse_directory(path: Optional[str] = None) -> Tuple[Path, Optional[Path], List[dict]]:
    if path:
        current = resolve_allowed_path(path)
    else:
        roots = _allowed_roots()
        if not roots:
            raise ValueError("未配置可浏览的数据集根目录")
        current = roots[0]

    if not current.is_dir():
        raise ValueError(f"不是目录: {current}")

    parent: Optional[Path] = None
    if current.parent != current:
        try:
            parent = resolve_allowed_path(str(current.parent))
        except ValueError:
            parent = None

    entries: List[dict] = []
    for entry in sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if entry.name.startswith("."):
            continue
        entries.append(
            {
                "name": entry.name,
                "path": str(entry.resolve()),
                "is_directory": entry.is_dir(),
            }
        )
    return current, parent, entries
