"""
XGBoost for Stock Market Prediction
Includes Optuna Hyperparamater Optimization
STOCKS = AAPL, MSFT, SPOT, TSLA, VTI, GOOGL
"""

from datetime import date, timedelta, datetime
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb

N_STEPS = 7  # Number of previous days to use for prediction
MODEL_SEED = 100

def read_data_from_file(csv):
    """
    Read stock data from csv file
    Inputs: csv 
    Returns: df
    """
    df = pd.read_csv(csv, usecols=['Date', 'Adj Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df

def process_dataframe(df, n_steps):
    """
    Process dataframe
    Inputs: df, n_steps
    Returns: df
    """
    df = df.copy()
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Adj Close'].shift(i)
    df.dropna(inplace=True)
    return df

def scale_dataframe(df):
    """
    Scale dataframe
    Inputs: df
    Returns: df_scaled, scaler
    """
    feature_cols = df.columns[1:]  # All columns except the target
    target_col = df.columns[0]  # The first column is the target

    scaler_features = MinMaxScaler(feature_range=(-1, 1))
    scaler_target = MinMaxScaler(feature_range=(-1, 1))

    df_features_scaled = scaler_features.fit_transform(df[feature_cols])
    df_target_scaled = scaler_target.fit_transform(df[[target_col]])

    df_scaled = np.hstack([df_target_scaled, df_features_scaled])
    return df_scaled, scaler_features, scaler_target

def split_data(scaled_data):
    """
    Split data
    Inputs: scaled_data
    Returns: X, y
    """
    X = scaled_data[:, 1:]  # All columns except the first one (target)
    y = scaled_data[:, 0]  # First column is the target
    return X, y

def split_data_into_train_test(X, y, split_ratio=0.95):
    """
    Split data into training and testing datasets
    Inputs: X, y, split ratio
    Returns: X_train, X_test, y_train, y_test
    """
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """
    Train and evaluate XGBoost model
    Inputs: X_train, y_train, X_test, y_test
    Returns: model, rmse, mape_value, y_test, y_pred
    """
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape_value = mean_absolute_percentage_error(y_test, y_pred)
    return model, rmse, mape_value, y_test, y_pred


def plot_results(y_test, y_pred):
    """
    Plot results
    Inputs: y_test, y_pred
    Returns: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred, label='Predictions')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def save_predictions(y_test, y_pred):
    """
    Save predictions in a csv file for further inspection
    Inputs: y_test, y_pred
    Returns: None
    """
    df = pd.DataFrame({'True Values': y_test, 'Predictions': y_pred})
    df.to_csv('predictions.csv', index=False)

def predict_next_day_close(model, df, scaler_features, scaler_target, n_steps):
    """
    Predict the next days Adj close
    Inputs: model, df, scaler, n_steps
    Returns: next_day_price
    """
    last_n_days = df['Adj Close'].values[-n_steps:]
    last_n_days_df = pd.DataFrame([last_n_days], columns=[f'Close(t-{i})' for i in range(n_steps, 0, -1)])
    last_n_days_df = last_n_days_df[scaler_features.feature_names_in_]
    last_n_days_scaled = scaler_features.transform(last_n_days_df)
    next_day_scaled = model.predict(last_n_days_scaled)  # Predict the next day's closing price in the scaled range
    next_day_price = scaler_target.inverse_transform(next_day_scaled.reshape(-1, 1)).flatten()[0] # Inverse transform to get the actual price
    return next_day_price

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna Hyperparamter Tuning
    Inputs: trial, X_train, y_train, X_val, y_val
    Returns: rmse
    """
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    return rmse

def predict_next_week_close(model, df, scaler_features, scaler_target, n_steps):
    """
    Predict Adj Close for next week
    Inputs: model, df, scaler_features, scaler_target, n_steps
    Returns: next_week_predictions
    """
    last_n_days = df['Adj Close'].values[-n_steps:]
    next_week_predictions = []

    for _ in range(7):
        last_n_days_df = pd.DataFrame([last_n_days], columns=[f'Close(t-{i})' for i in range(n_steps, 0, -1)])
        last_n_days_df = last_n_days_df[scaler_features.feature_names_in_]
        last_n_days_scaled = scaler_features.transform(last_n_days_df)

        next_day_scaled = model.predict(last_n_days_scaled)
        next_day_price = scaler_target.inverse_transform(next_day_scaled.reshape(-1, 1)).flatten()[0]
        next_week_predictions.append(next_day_price)
        last_n_days = np.append(last_n_days[1:], next_day_price)
    return next_week_predictions

def main():
    """
    Main program function
    """
df = read_data_from_file('Stock Data/AAPL.csv')
    processed_df = process_dataframe(df, N_STEPS)
    scaled_data, scaler_features, scaler_target = scale_dataframe(processed_df)
    X, y = split_data(scaled_data)
    X_train, X_test, y_train, y_test = split_data_into_train_test(X, y)

    # Split train into train and validation sets for Optuna optimization
    split_index = int(len(X_train) * 0.8)
    X_train_split, X_val_split = X_train[:split_index], X_train[split_index:]
    y_train_split, y_val_split = y_train[:split_index], y_train[split_index:]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_split, y_train_split, X_val_split, y_val_split), n_trials=30)
    print('Best hyperparameters:', study.best_params)
    print('Best RMSE:', study.best_value)

    # Use best hyperparameters to train the final model
    best_params = study.best_params
    best_params.update({
        "objective": "reg:squarederror",
        "verbosity": 0,
        "seed": MODEL_SEED
    })

    model, rmse, mape_value, y_test, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, best_params)
    print(f'XGBoost RMSE: {rmse}')
    print(f'XGBoost MAPE: {mape_value}%')
    # plot_results(y_test, y_pred)  # Uncomment for visualization
    save_predictions(y_test, y_pred)
    
    next_day_price = predict_next_day_close(model, df, scaler_features, scaler_target, N_STEPS)
    print(f'Predicted next day adj closing price: {next_day_price}')
    next_week_prices = predict_next_week_close(model, df, scaler_features, scaler_target, N_STEPS)
    print(f'Predicted next week adj closing prices: {next_week_prices}')

if __name__ == "__main__":
    main()
