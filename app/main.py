from fastapi import FastAPI
from app.data_fetcher import fetch_crypto_data
from app.model import train_model, predict_price

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Crypto Price Predictor is up!"}

@app.get("/predict")
def get_prediction():
    data = fetch_crypto_data()
    model, last_features = train_model(data)
    prediction = predict_price(model, last_features)
    return {"prediction": prediction}
