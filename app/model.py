import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

def train_model(data: pd.DataFrame, model_type='linear'):
    """
    Trains a model on the provided enriched data.
    :param data: Enriched dataframe with features.
    :param model_type: 'linear' for Linear Regression, 'rf' for Random Forest.
    :return: Trained model, last features vector, training RMSE, and testing RMSE.
    """
    data = data.copy()
    # Use the features we engineered in the data fetcher
    features = ['MA_7', 'MA_30', 'Daily_Return', 'Volatility_7']
    X = data[features]
    y = data['price']
    
    # Split data into training and testing sets (80/20 split, no shuffling for time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if model_type == 'rf':
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestRegressor(random_state=42)
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print("Best parameters for RF:", grid_search.best_params_)
    else:
        # Use a simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Use the last row of the training set's features for prediction (for demonstration)
    last_features = X_train.iloc[-1].values.reshape(1, -1)
    
    return model, last_features, train_rmse, test_rmse



def predict_price(model, last_features, future_days: int = 1):
    """
    Predicts the price using the last available feature vector.
    For now, we ignore future_days and simply predict with the last row.
    """
    return model.predict(last_features)[0]
