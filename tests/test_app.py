from fastapi.testclient import TestClient
from app.main import app
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Crypto Price Predictor is up!"}

def test_predict_linear():
    response = client.get("/predict?model_type=linear")
    json_data = response.json()
    assert response.status_code == 200
    # Ensure response contains expected keys
    for key in ["prediction", "training_rmse", "testing_rmse", "model_type"]:
        assert key in json_data
    assert json_data["model_type"] == "linear"

def test_predict_rf():
    response = client.get("/predict?model_type=rf")
    json_data = response.json()
    assert response.status_code == 200
    for key in ["prediction", "training_rmse", "testing_rmse", "model_type"]:
        assert key in json_data
    assert json_data["model_type"] == "rf"
