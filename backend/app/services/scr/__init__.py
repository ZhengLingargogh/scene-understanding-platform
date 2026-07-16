from app.services.scr.base import SCRModel
from app.services.scr.mock import MockSCRModel

__all__ = ["SCRModel", "AceSCRModel", "MockSCRModel", "get_shared_ace_scr"]


def __getattr__(name: str):
    if name == "AceSCRModel":
        from app.services.scr.ace import AceSCRModel

        return AceSCRModel
    if name == "get_shared_ace_scr":
        from app.services.scr.ace import get_shared_ace_scr

        return get_shared_ace_scr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
