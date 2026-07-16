"""Benchmark jobs orchestrated via FeatureExtractor plugins."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional
from uuid import uuid4

from app.compat import UTC, datetime
from app.services.plugins import get_feature_extractor, get_semantic_segmenter

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def create_feature_extraction_job(
    model_id: str,
    dataset_id: str,
    input_path: str,
    output_path: str,
) -> Dict[str, Any]:
    job_id = str(uuid4())
    now = datetime.now(UTC)
    job = {
        "job_id": job_id,
        "type": "feature_extraction",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "input_path": input_path,
        "output_path": output_path,
        "status": "running",
        "progress": 0,
        "total": 0,
        "processed": 0,
        "message": "Extracting features…",
        "created_at": now.isoformat(),
        "completed_at": None,
    }
    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_feature_extraction,
        args=(job_id, model_id, dataset_id, input_path, output_path),
        daemon=True,
    )
    thread.start()
    return dict(job)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _run_feature_extraction(
    job_id: str,
    model_id: str,
    dataset_id: str,
    input_path: str,
    output_path: str,
) -> None:
    extractor = get_feature_extractor(model_id)

    def on_progress(processed: int, total: int, message: str) -> None:
        progress = int(processed * 100 / total) if total else 0
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["processed"] = processed
            job["total"] = total
            job["progress"] = progress
            job["message"] = message

    try:
        summary = extractor.infer_dataset(
            dataset_id=dataset_id,
            input_path=input_path,
            output_path=output_path,
            on_progress=on_progress,
        )
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = summary.get("status", "completed")
            job["progress"] = 100
            job["total"] = summary.get("total", job.get("total", 0))
            job["processed"] = job["total"]
            job["message"] = summary.get("message", "Feature extraction completed")
            job["completed_at"] = datetime.now(UTC).isoformat()
    except Exception as exc:
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = "failed"
            job["message"] = str(exc)
            job["completed_at"] = datetime.now(UTC).isoformat()


def create_semantic_segmentation_job(
    model_id: str,
    dataset_id: str,
    input_path: str,
    output_path: str,
) -> Dict[str, Any]:
    job_id = str(uuid4())
    now = datetime.now(UTC)
    job = {
        "job_id": job_id,
        "type": "semantic_segmentation",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "input_path": input_path,
        "output_path": output_path,
        "status": "running",
        "progress": 0,
        "total": 0,
        "processed": 0,
        "message": "语义分割任务排队中…",
        "created_at": now.isoformat(),
        "completed_at": None,
    }
    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_semantic_segmentation,
        args=(job_id, model_id, dataset_id, input_path, output_path),
        daemon=True,
    )
    thread.start()
    return dict(job)


def _run_semantic_segmentation(
    job_id: str,
    model_id: str,
    dataset_id: str,
    input_path: str,
    output_path: str,
) -> None:
    segmenter = get_semantic_segmenter(model_id)

    def on_progress(processed: int, total: int, message: str) -> None:
        progress = int(processed * 100 / total) if total else 0
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["processed"] = processed
            job["total"] = total
            job["progress"] = progress
            job["message"] = message

    try:
        summary = segmenter.infer_dataset(
            dataset_id=dataset_id,
            input_path=input_path,
            output_path=output_path,
            on_progress=on_progress,
        )
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = summary.get("status", "completed")
            job["progress"] = 100
            job["total"] = summary.get("total", job.get("total", 0))
            job["processed"] = job["total"]
            job["message"] = summary.get("message", "Semantic segmentation completed")
            job["completed_at"] = datetime.now(UTC).isoformat()
    except Exception as exc:
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = "failed"
            job["message"] = str(exc)
            job["completed_at"] = datetime.now(UTC).isoformat()
