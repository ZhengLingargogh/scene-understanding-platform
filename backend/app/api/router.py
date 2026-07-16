from fastapi import APIRouter

from app.api.v1 import benchmark, inference, media, models, scenes

api_router = APIRouter()

api_router.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
# Backward-compatible alias
api_router.include_router(inference.router, prefix="/localization", tags=["localization (deprecated)"])
api_router.include_router(benchmark.router, prefix="/benchmarks", tags=["benchmarks"])
