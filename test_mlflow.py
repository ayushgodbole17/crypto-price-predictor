from app.data_fetcher import fetch_crypto_data
from app.model import train_model, load_model, predict_price

data = fetch_crypto_data(days=90)
model, last_features, train_rmse, test_rmse = train_model(data, model_type='linear')
print("Train RMSE:", train_rmse, "Test RMSE:", test_rmse)
loaded_model = load_model()
prediction = predict_price(loaded_model, last_features)
print("Prediction:", prediction)
