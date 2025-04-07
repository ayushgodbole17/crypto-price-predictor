import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import mlflow


def train_model(data: pd.DataFrame, model_type='linear'):
    """
    Trains a model on the provided enriched data, logs parameters and metrics with MLflow,
    and saves the model as an artifact.
    
    :param data: Enriched dataframe with features (must include 'price' and e.g. 'MA_7', 'MA_30', etc.).
    :param model_type: 'linear' for Linear Regression, 'rf' for Random Forest.
    :return: Trained model, last feature vector (for demonstration), training RMSE, and testing RMSE.
    """
    data = data.copy()
    features = ['MA_7', 'MA_30', 'Daily_Return', 'Volatility_7']
    
    # Ensure the required columns exist
    if not all(f in data.columns for f in features + ['price']):
        raise ValueError(f"Data is missing required columns: {features + ['price']}")

    X = data[features]
    y = data['price']
    
    # Split data into training and testing sets (80/20), shuffle=False to respect time order
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Start an MLflow run
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_samples", len(data))
    
        if model_type == 'rf':
            # Use TimeSeriesSplit for time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            }
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf,
                param_grid,
                cv=tscv,                     # Time-series aware CV
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            # Log the best hyperparameters
            mlflow.log_params(grid_search.best_params_)
            print("Best parameters for RF:", grid_search.best_params_)
        else:
            # Train a simple Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
    
        # Evaluate performance
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
    
        # Save the model locally and log as artifact
        model_file = "model.pkl"
        joblib.dump(model, model_file)
        mlflow.log_artifact(model_file)
    
        # For demonstration, get the last feature vector from training data
        last_features = X_train.iloc[-1].values.reshape(1, -1)
    
        return model, last_features, train_rmse, test_rmse


def predict_price(model, last_features, future_days: int = 1):
    """
    Predicts the price using the provided feature vector.
    
    :param model: Trained model ready for inference.
    :param last_features: 2D array representing the last row of features (or any new feature vector).
    :param future_days: (Optional) Not used in direct inference; kept for interface consistency.
    :return: A single float prediction from the model.
    """
    return model.predict(last_features)[0]


def load_model(filename="model.pkl"):
    """
    Loads a saved model from disk.
    
    :param filename: Path to the model file.
    :return: The loaded model object.
    """
    return joblib.load(filename)
