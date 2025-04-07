import os
import logging
from fastapi import FastAPI, Query
from app.data_fetcher import fetch_crypto_data
from app.model import load_model, predict_price, train_model
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from fastapi.staticfiles import StaticFiles

# Mount the static directory so that index.html is served at the root URL


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="static")

# Expose Prometheus metrics
Instrumentator().instrument(app).expose(app)
prediction_counter = Counter("model_predictions_total", "Total model predictions", ["model_type"])

MODEL_FILE = "model.pkl"

# At startup, load a pre-trained model if available; otherwise, train one.
if os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE)
    logger.info("Loaded pre-trained model.")
else:
    logger.info("No pre-trained model found. Training a new model...")
    data = fetch_crypto_data(days=90)
    model, last_features, train_rmse, test_rmse = train_model(data, model_type="linear")
    logger.info("Trained and saved new model.")

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Crypto Price Predictor is up!"}

@app.get("/predict")
def get_prediction(model_type: str = Query("linear", enum=["linear", "rf"])):
    logger.info(f"Predict endpoint accessed with model_type={model_type}")
    try:
        # Use the pre-loaded model for prediction.
        data = fetch_crypto_data(days=90)
        features = ['MA_7', 'MA_30', 'Daily_Return', 'Volatility_7']
        last_features = data[features].iloc[-1].values.reshape(1, -1)
        prediction = predict_price(model, last_features)
        prediction_counter.labels(model_type=model_type).inc()
        logger.info(f"Prediction made: {prediction}")
        return {
            "prediction": prediction,
            "model_type": model_type
        }
    except Exception as e:
        logger.error("Error during prediction", exc_info=True)
        return {"error": str(e)}

@app.post("/retrain")
def retrain():
    logger.info("Retrain endpoint accessed")
    try:
        data = fetch_crypto_data(days=90)
        new_model, last_features, train_rmse, test_rmse = train_model(data, model_type="linear")
        global model
        model = new_model  # Update the global model with the retrained one.
        logger.info("Model retrained and updated.")
        return {
            "message": "Model retrained successfully.",
            "training_rmse": train_rmse,
            "testing_rmse": test_rmse
        }
    except Exception as e:
        logger.error("Error during retraining", exc_info=True)
        return {"error": str(e)}
