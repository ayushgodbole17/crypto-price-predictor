import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def train_model(data: pd.DataFrame):
    data = data.copy()
    data['timestamp'] = np.arange(len(data))  # Replacing real timestamps with indices
    X = data[['timestamp']]
    y = data['price']
    model = LinearRegression().fit(X, y)
    return model, len(data)  # Return model + last timestamp index

def predict_price(model, last_index: int, future_days: int = 1):
    future_x = [[last_index + future_days]]
    return model.predict(future_x)[0]
