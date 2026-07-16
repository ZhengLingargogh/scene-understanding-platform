"""Mock retriever for Inference (no real embeddings or Faiss index)."""

from __future__ import annotations

from typing import List, Optional

from app.schemas.pipeline import ImageRetrievalResult, RetrievalReference
from app.services.mock_datasets import get_mock_dataset
from app.services.retrievers.base import Retriever

DEFAULT_TOP_K = 8


class MockRetriever(Retriever):
    """
    Placeholder retriever — returns deterministic top-K references.

    Future implementations: SALADRetriever, NetVLADRetriever, …
    """

    def __init__(self, model_id: str = "mock-retriever") -> None:
        self._model_id = model_id
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(
        self,
        *,
        image_path: str,
        dataset_id: str,
        top_k: int = DEFAULT_TOP_K,
        scene_id: Optional[str] = None,
    ) -> ImageRetrievalResult:
        if not self._loaded:
            self.load()

        dataset = get_mock_dataset(dataset_id)
        dataset_name = dataset["name"] if dataset else dataset_id
        total = int(dataset["image_count"]) if dataset else 256

        references: List[RetrievalReference] = []
        for rank in range(top_k):
            frame_idx = (rank * 17 + 3) % max(total, top_k)
            score = round(0.98 - rank * 0.045, 3)
            references.append(
                RetrievalReference(
                    id="{}-rank-{}".format(dataset_id, rank + 1),
                    label="{} · frame_{:04d}.jpg".format(dataset_name, frame_idx),
                    score=max(score, 0.1),
                    image_path="datasets/{}/rgb/frame_{:04d}.jpg".format(dataset_id, frame_idx),
                )
            )

        return ImageRetrievalResult(
            status="completed",
            message="Mock retrieval via '{}' on dataset '{}' (query: {})".format(
                self._model_id, dataset_id, image_path
            ),
            top_k=top_k,
            dataset_id=dataset_id,
            references=references,
        )
