from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_benchmark_service
from app.schemas.benchmark import (
    BenchmarkCreate,
    BenchmarkResponse,
    FeatureExtractionJobResponse,
    FeatureExtractionRunRequest,
    SemanticSegmentationJobResponse,
    SemanticSegmentationRunRequest,
)
from app.schemas.pipeline import BENCHMARK_PIPELINE_STAGES, PIPELINE_STAGE_LABELS, PipelineStage
from app.services.plugins import registry
from app.services.benchmark_jobs import (
    create_feature_extraction_job,
    create_semantic_segmentation_job,
    get_job,
)
from app.services.benchmark_service import BenchmarkService
from app.services.mock_datasets import list_mock_datasets, resolve_input_path

router = APIRouter()


@router.get("/plugins")
async def list_benchmark_plugins():
    return registry.list_plugins()


@router.get("/datasets")
async def list_benchmark_datasets():
    """Mock datasets for benchmark feature extraction."""
    return list_mock_datasets()


@router.get("/pipeline-stages")
async def list_benchmark_pipeline_stages():
    return [
        {"stage": stage.value, "label": PIPELINE_STAGE_LABELS[stage]}
        for stage in BENCHMARK_PIPELINE_STAGES
    ]


@router.get("", response_model=List[BenchmarkResponse])
async def list_benchmarks(service: BenchmarkService = Depends(get_benchmark_service)):
    return service.list_benchmarks()


@router.post("", response_model=BenchmarkResponse, status_code=status.HTTP_201_CREATED)
async def create_benchmark(
    payload: BenchmarkCreate,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    return service.create_benchmark(payload)


@router.post("/feature-extraction/run", response_model=FeatureExtractionJobResponse)
async def run_feature_extraction(payload: FeatureExtractionRunRequest):
    """
    Start feature extraction via ``FeatureExtractor`` plugin.

    Poll ``GET /benchmarks/feature-extraction/{job_id}`` for progress (0–100).
    """
    try:
        input_path = resolve_input_path(payload.dataset_id, payload.input_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    job = create_feature_extraction_job(
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        input_path=input_path,
        output_path=payload.output_path,
    )
    return FeatureExtractionJobResponse(**job)


@router.get("/feature-extraction/{job_id}", response_model=FeatureExtractionJobResponse)
async def get_feature_extraction_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return FeatureExtractionJobResponse(**job)


@router.post("/semantic-segmentation/run", response_model=SemanticSegmentationJobResponse)
async def run_semantic_segmentation(payload: SemanticSegmentationRunRequest):
    """
    Start semantic segmentation via ``SemanticSegmenter`` plugin (SAM).

    Poll ``GET /benchmarks/semantic-segmentation/{job_id}`` for progress (0–100).
    """
    try:
        input_path = resolve_input_path(payload.dataset_id, payload.input_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    job = create_semantic_segmentation_job(
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        input_path=input_path,
        output_path=payload.output_path,
    )
    return SemanticSegmentationJobResponse(**job)


@router.get("/semantic-segmentation/{job_id}", response_model=SemanticSegmentationJobResponse)
async def get_semantic_segmentation_job(job_id: str):
    job = get_job(job_id)
    if job is None or job.get("type") != "semantic_segmentation":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return SemanticSegmentationJobResponse(**job)


@router.get("/{benchmark_id}", response_model=BenchmarkResponse)
async def get_benchmark(
    benchmark_id: str,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    benchmark = service.get_benchmark(benchmark_id)
    if benchmark is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark not found")
    return benchmark


@router.delete("/{benchmark_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_benchmark(
    benchmark_id: str,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    if not service.delete_benchmark(benchmark_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark not found")
