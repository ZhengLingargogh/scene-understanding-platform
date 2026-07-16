"""Deprecated: use app.services.retrievers.mock.MockRetriever via plugin registry."""

from app.services.retrievers.mock import DEFAULT_TOP_K as RETRIEVAL_TOP_K
from app.services.plugins import get_retriever


def mock_retrieve_images(dataset_id: str, model_id: str = "ace", top_k: int = RETRIEVAL_TOP_K):
    retriever = get_retriever(model_id)
    return retriever.infer(
        image_path="",
        dataset_id=dataset_id,
        top_k=top_k,
    )
