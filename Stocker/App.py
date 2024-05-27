import flask
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
from XGBoost import StockPredictorXGBoost
from BNN import StockPredictorBNN, BNN

def objective(trial, stock_predictor):
    # Define the hyperparameters to tune
    hid_dim = trial.suggest_int('hid_dim', 5, 50)
    prior_scale = trial.suggest_float('prior_scale', 1.0, 20.0)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    model = BNN(in_dim=7, hid_dim=hid_dim, prior_scale=prior_scale)
    guide = stock_predictor.get_guide(model)
    stock_predictor.train(model, guide, num_iterations=1000, lr=lr)
    
    # Evaluate the model on the validation set
    y_pred_mean, _ = stock_predictor.predict(model, guide, stock_predictor.X_test_tensor)
    y_pred_mean = torch.tensor(y_pred_mean)
    rmse = torch.sqrt(torch.mean((y_pred_mean - stock_predictor.y_test_tensor) ** 2))
    return rmse.item()

predictor = StockPredictorXGBoost('Stock Data/AAPL.csv')
best_params = predictor.optimize_hyperparameters(n_trials=30)
model = predictor.train_with_optimal_hyperparameters(best_params)
next_day_price = predictor.predict_next_day_close(model)
print(f'Predicted next day adj closing price: {next_day_price}')
next_week_prices = predictor.predict_next_week_close(model)
print(f'Predicted next week adj closing prices: {next_week_prices}')

stock_predictor = StockPredictorBNN('Stock Data/AAPL.csv')
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('Best trial:')
trial = study.best_trial
print(f'  RMSE: {trial.value}')
print('  Best hyperparameters:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
# Train the model with the best hyperparameters
best_hid_dim = trial.params['hid_dim']
best_prior_scale = trial.params['prior_scale']
best_lr = trial.params['lr']
best_model = BNN(in_dim=7, hid_dim=best_hid_dim, prior_scale=best_prior_scale)
best_guide = stock_predictor.get_guide(best_model)
best_svi = stock_predictor.train(best_model, best_guide, num_iterations=1000, lr=best_lr)
# Predict on the test set and plot the results
y_pred_mean, y_pred_std = stock_predictor.predict(best_model, best_guide, stock_predictor.X_test_tensor)
stock_predictor.plot_results(stock_predictor.y_test, y_pred_mean, y_pred_std)
# Predict the next day adjusted closing price and next week adjusted closing prices
next_day_price = stock_predictor.predict_next_day_close(best_model, best_guide)
print(f'Predicted next day adjusted closing price: {next_day_price}')
next_week_prices = stock_predictor.predict_next_week_close(best_model, best_guide)
print(f'Predicted next week adjusted closing prices: {next_week_prices}')
