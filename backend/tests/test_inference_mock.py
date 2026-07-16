from app.services.plugins import get_feature_extractor, get_retriever, get_scr_model, registry
from app.services.retrievers import DEFAULT_TOP_K


def test_mock_retriever_via_registry_returns_top_8():
    retriever = get_retriever("ace")
    result = retriever.infer(
        image_path="/tmp/query.jpg",
        dataset_id="crossloc-nature-test",
        top_k=DEFAULT_TOP_K,
    )
    assert result.status == "completed"
    assert result.top_k == DEFAULT_TOP_K
    assert len(result.references) == 8
    assert result.references[0].score >= result.references[1].score


def test_plugin_registry_lists_backends():
    plugins = registry.list_plugins()
    assert "ace" in plugins["retrievers"]
    assert "salad" in plugins["retrievers"]
    assert "mock-extractor" in plugins["feature_extractors"]
    assert "salad" in plugins["feature_extractors"]
    assert "ace" in plugins["scr_models"]


def test_mock_scr_infer():
    scr = get_scr_model("mock-scr")
    result = scr.infer(image_path="/tmp/q.jpg", scene_id="nature")
    assert result["pose_matrix"] is not None
    assert result["pose_backend"] == "mock"


def test_inference_datasets_endpoint():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/inference/datasets")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert "id" in data[0]


def test_feature_extraction_job_lifecycle():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/benchmarks/feature-extraction/run",
        json={
            "model_id": "mock-extractor",
            "dataset_id": "crossloc-nature-train",
            "input_path": "/tmp/in",
            "output_path": "/tmp/out",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "running"

    poll = client.get("/api/v1/benchmarks/feature-extraction/{}".format(job_id))
    assert poll.status_code == 200
    assert "progress" in poll.json()
