from uuid import uuid4

from app.compat import UTC, datetime
from app.schemas.benchmark import BenchmarkCreate, BenchmarkResponse
from app.schemas.pipeline import BENCHMARK_PIPELINE_STAGES


class BenchmarkService:
    """Dataset-wide benchmark jobs (evaluation logic pending)."""

    def __init__(self) -> None:
        self._benchmarks = {}  # type: dict

    def list_benchmarks(self):
        return list(self._benchmarks.values())

    def get_benchmark(self, benchmark_id: str):
        return self._benchmarks.get(benchmark_id)

    def create_benchmark(self, payload: BenchmarkCreate) -> BenchmarkResponse:
        now = datetime.now(UTC)
        benchmark = BenchmarkResponse(
            id=str(uuid4()),
            name=payload.name,
            scene_id=payload.scene_id,
            model_id=payload.model_id,
            dataset_id=payload.dataset_id,
            dataset_path=payload.dataset_path,
            input_path=payload.input_path,
            output_path=payload.output_path,
            pipeline_stages=payload.pipeline_stages or list(BENCHMARK_PIPELINE_STAGES),
            status="pending",
            metrics=None,
            created_at=now,
            updated_at=now,
        )
        self._benchmarks[benchmark.id] = benchmark
        return benchmark

    def delete_benchmark(self, benchmark_id: str) -> bool:
        return self._benchmarks.pop(benchmark_id, None) is not None
