"""API contract tests. They run without a trained model on disk."""


def test_health_is_always_reachable(client):
    """Health must answer even when no model has been trained yet."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_an_empty_batch(client):
    """Input validation happens before the model is ever touched."""
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 422
