"""Interactive segmentation session store and point-prompt inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from PIL import Image

from app.schemas.segmentation import SegmentationPredictResponse, SegmentationSessionResponse
from app.services.plugins.registry import get_segmenter

SESSION_TTL_SECONDS = 3600


@dataclass
class SegmentationSession:
    session_id: str
    image_path: str
    image_width: int
    image_height: int
    model_id: str
    created_at: float


class InteractiveSegmentationService:
    def __init__(self) -> None:
        self._sessions: Dict[str, SegmentationSession] = {}

    def create_session(self, *, image_path: str, model_id: str) -> SegmentationSessionResponse:
        self._purge_expired()
        path = Path(image_path)
        with Image.open(path) as img:
            width, height = img.size

        session_id = str(uuid4())
        self._sessions[session_id] = SegmentationSession(
            session_id=session_id,
            image_path=str(path),
            image_width=width,
            image_height=height,
            model_id=model_id,
            created_at=time.time(),
        )

        segmenter = get_segmenter(model_id)
        backend = segmenter.model_id
        if segmenter.is_implemented and hasattr(segmenter, "prepare_image"):
            segmenter.prepare_image(path)  # type: ignore[attr-defined]
            message = "Interactive session ready ({})".format(backend)
        elif segmenter.is_implemented:
            message = "Interactive session ready ({})".format(backend)
        else:
            message = "Session ready — {} is reserved; predict will fail until wired".format(model_id)

        return SegmentationSessionResponse(
            session_id=session_id,
            model_id=model_id,
            image_width=width,
            image_height=height,
            backend=backend,
            message=message,
        )

    def predict(
        self,
        *,
        session_id: str,
        point_x: float,
        point_y: float,
    ) -> SegmentationPredictResponse:
        self._purge_expired()
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Segmentation session not found: {}".format(session_id))

        segmenter = get_segmenter(session.model_id)
        result = segmenter.segment_point(
            image_path=Path(session.image_path),
            point_x=point_x,
            point_y=point_y,
            image_width=session.image_width,
            image_height=session.image_height,
        )

        mask_flat = result.mask.astype(int).reshape(-1).tolist()
        return SegmentationPredictResponse(
            session_id=session_id,
            model_id=session.model_id,
            point=[point_x, point_y],
            image_width=session.image_width,
            image_height=session.image_height,
            mask=mask_flat,
            contour=result.contour,
            score=result.score,
            backend=result.backend,
            message=result.message,
        )

    def delete_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if now - session.created_at > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]


_segmentation_service: Optional[InteractiveSegmentationService] = None


def get_interactive_segmentation_service() -> InteractiveSegmentationService:
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = InteractiveSegmentationService()
    return _segmentation_service
