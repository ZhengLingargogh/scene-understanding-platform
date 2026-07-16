from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_scenes_empty():
    response = client.get("/api/v1/scenes")
    assert response.status_code == 200
    assert response.json() == []
