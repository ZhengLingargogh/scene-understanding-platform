from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.db.session import SessionLocal, init_db
from app.services.dataset_catalog import invalidate_family_cache
from app.services.dataset_registry import load_custom_families
from app.services.mock_datasets import rebuild_mock_datasets_index
from app.services.scene_seed import seed_catalog_scenes


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    load_custom_families()
    invalidate_family_cache()
    rebuild_mock_datasets_index()
    with SessionLocal() as db:
        seed_catalog_scenes(db)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.app_version}
