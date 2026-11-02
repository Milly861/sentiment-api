import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_positive(client):
    response = client.post("/predict", json={"text": "I love this movie, it is great"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= data["score"] <= 1.0


def test_predict_missing_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422