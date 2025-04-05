import logging
from fastapi import FastAPI, Query
from app.data_fetcher import fetch_crypto_data
from app.model import train_model, predict_price

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Crypto Price Predictor is up!"}

@app.get("/predict")
def get_prediction(model_type: str = Query("linear", enum=["linear", "rf"])):
    logger.info(f"Predict endpoint accessed with model_type={model_type}")
    try:
        data = fetch_crypto_data(days=90)
        model, last_features, train_rmse, test_rmse = train_model(data, model_type=model_type)
        prediction = predict_price(model, last_features)
        logger.info(f"Prediction made: {prediction}")
        return {
            "prediction": prediction,
            "training_rmse": train_rmse,
            "testing_rmse": test_rmse,
            "model_type": model_type
        }
    except Exception as e:
        logger.error("Error during prediction", exc_info=True)
        return {"error": str(e)}
