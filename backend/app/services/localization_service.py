"""Backward-compatible helpers for ACE localization."""

from functools import lru_cache

from app.services.localization import LocalizationService
from app.services.scr.ace import get_shared_ace_scr


@lru_cache(maxsize=1)
def get_ace_localization_service() -> LocalizationService:
    """Return the underlying ACE LocalizationService engine."""
    scr = get_shared_ace_scr()
    if scr._engine is None:
        scr.load()
    assert scr._engine is not None
    return scr._engine


# Re-export for legacy imports
__all__ = ["get_ace_localization_service", "LocalizationService"]
