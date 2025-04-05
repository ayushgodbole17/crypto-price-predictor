from fastapi import FastAPI, Query
from app.data_fetcher import fetch_crypto_data
from app.model import train_model, predict_price

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Crypto Price Predictor is up!"}

@app.get("/predict")
def get_prediction(model_type: str = Query("linear", enum=["linear", "rf"])):
    """
    Predicts the crypto price.
    Query parameter 'model_type' can be 'linear' or 'rf' for Random Forest.
    """
    data = fetch_crypto_data()
    model, last_features, rmse = train_model(data, model_type=model_type)
    prediction = predict_price(model, last_features)
    return {"prediction": prediction, "training_rmse": rmse, "model_type": model_type}
