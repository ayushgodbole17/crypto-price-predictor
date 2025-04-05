import pandas as pd
from app.model import train_model, predict_price

def test_train_model_linear():
    # Mock data
    data = pd.DataFrame({
        'price': [100, 102, 105, 103, 106, 107, 108, 110, 111, 115,
                  116, 117, 118, 119, 120, 121, 122, 124, 125, 126,
                  127, 128, 129, 130, 131, 132, 133, 134, 135, 136],
        'MA_7': [100]*30,
        'MA_30': [100]*30,
        'Daily_Return': [0.01]*30,
        'Volatility_7': [0.02]*30,
    })
    
    model, last_features, train_rmse, test_rmse = train_model(data, model_type='linear')
    
    assert model is not None
    assert last_features.shape == (1, 4)
    assert isinstance(train_rmse, float)
    assert isinstance(test_rmse, float)

def test_predict_price():
    from sklearn.linear_model import LinearRegression
    import numpy as np
    model = LinearRegression().fit([[0], [1], [2]], [10, 20, 30])
    prediction = predict_price(model, [[3]])
    assert isinstance(prediction, float)
