import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def train_model(data: pd.DataFrame):
    data = data.copy()
    # Define the features to use
    features = ['MA_7', 'MA_30', 'Daily_Return', 'Volatility_7']
    X = data[features]
    y = data['price']
    model = LinearRegression().fit(X, y)
    # Get the last row's feature values to use for prediction
    last_features = data.iloc[-1][features].values.reshape(1, -1)
    return model, last_features

def predict_price(model, last_features, future_days: int = 1):
    # For now, we ignore future_days and simply predict using the last available features.
    return model.predict(last_features)[0]
