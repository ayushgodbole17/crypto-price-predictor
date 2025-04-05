import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def train_model(data: pd.DataFrame, model_type='linear'):
    """
    Trains a model on the provided data.
    :param data: Enriched dataframe with features.
    :param model_type: 'linear' for Linear Regression, 'rf' for Random Forest.
    :return: Trained model, last features vector, and RMSE on training data.
    """
    # Define the features to use (these should be added by your data enrichment code)
    features = ['MA_7', 'MA_30', 'Daily_Return', 'Volatility_7']
    X = data[features]
    y = data['price']
    
    if model_type == 'rf':
        # Use Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        # Default to Linear Regression
        model = LinearRegression()
        
    model.fit(X, y)
    
    # Predict on training data for evaluation
    y_pred = model.predict(X)
    rmse = sqrt(mean_squared_error(y, y_pred))
    
    # Get the last row's feature values for prediction (reshaped as needed)
    last_features = data.iloc[-1][features].values.reshape(1, -1)
    
    return model, last_features, rmse

def predict_price(model, last_features, future_days: int = 1):
    """
    Predicts the price using the last available feature vector.
    For now, we ignore future_days and simply predict with the last row.
    """
    return model.predict(last_features)[0]
