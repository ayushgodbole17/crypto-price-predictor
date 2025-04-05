import pandas as pd
from app.model import train_model, predict_price

def test_train_model_linear():
    # Create a simple DataFrame with dummy data
    data = pd.DataFrame({
        'price': [100, 102, 105, 103, 106, 107, 108, 110, 111, 115],
        'MA_7': [100]*10,
        'MA_30': [100]*10,
        'Daily_Return': [0.01]*10,
        'Volatility_7': [0.02]*10,
    })
    
    model, last_features, train_rmse, test_rmse = train_model(data, model_type='linear')
    
    # Check that the model is not None and metrics are computed
    assert model is not None
    assert last_features.shape == (1, 4)
    assert isinstance(train_rmse, float)
    assert isinstance(test_rmse, float)

def test_predict_price():
    # Use a simple linear model on dummy data
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit([[0], [1], [2]], [10, 20, 30])
    prediction = predict_price(model, [[3]])
    assert isinstance(prediction, float)
