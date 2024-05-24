"""
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

N_STEPS = 7  # Number of previous days to use for prediction
N_ESTIMATORS = 100  # Number of estimators
MAX_DEPTH = 3  # Max tree depth
LEARNING_RATE = 0.1  # Learning rate
MIN_CHILD_WEIGHT = 1  # Minimum sum of instance weight
SUBSAMPLE = 1  # Subsample ratio of training instance
COLSAMPLE_BYTREE = 1  # Subsample ratio of columns when constructing each tree
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
    feature_cols = df.columns[1:]
    target_col = df.columns[0]
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
    X = scaled_data[:, 1:]  
    y = scaled_data[:, 0]  
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

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train and evaluate XGBoost model
    Inputs: X_train, y_train, X_test, y_test
    Returns: model, rmse, mape_value, y_test, y_pred
    """
    model = XGBRegressor(
        seed=MODEL_SEED,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_child_weight=MIN_CHILD_WEIGHT,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE
    )
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

def main():
    """
    Main program function
    """
    df = read_data_from_file('Stock Data/AAPL.csv')
    processed_df = process_dataframe(df, N_STEPS)
    scaled_data, scaler_features, scaler_target = scale_dataframe(processed_df)
    X, y = split_data(scaled_data)
    X_train, X_test, y_train, y_test = split_data_into_train_test(X, y)
    model, rmse, mape_value, y_test, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f'XGBoost RMSE: {rmse}')
    print(f'XGBoost MAPE: {mape_value}%')
    plot_results(y_test, y_pred)
    save_predictions(y_test, y_pred)
    
    next_day_price = predict_next_day_close(model, df, scaler_features, scaler_target, N_STEPS)
    print(f'Predicted next day adj closing price: {next_day_price}')

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
