"""
BNN for Stock Prediction
Includes Optuna Hyperparamater Optimization
STOCKS = AAPL, MSFT, SPOT, TSLA, VTI, GOOGL
"""

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
import matplotlib

matplotlib.use('Agg')

class BNN(PyroModule):
    """
    BNN class
    """
    def __init__(self, in_dim=7, out_dim=1, hid_dim=10, prior_scale=10):
        super().__init__()
        self.activation = nn.Tanh()
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = PyroModule[nn.Linear](hid_dim, out_dim)

        self.layer1.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(0., prior_scale).expand([out_dim, hid_dim]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(0., prior_scale).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(2.0, 1.0))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu

class StockPredictorBNN:
    """
    Stock Predictor Class
    """
    def __init__(self, csv_file, n_steps=7):
        self.n_steps = n_steps
        self.df = self.read_data_from_file(csv_file)
        self.scaler_features, self.scaler_target = MinMaxScaler(), MinMaxScaler()
        self.processed_df = self.process_dataframe(self.df, n_steps)
        self.scaled_data = self.scale_dataframe(self.processed_df)
        self.X, self.y = self.split_data(self.scaled_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data_into_train_test(self.X, self.y)
        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

    def read_data_from_file(self, csv):
        """
        Read stockdata from csv
        Input: csv 
        Returns: df
        """
        df = pd.read_csv(csv, usecols=['Date', 'Adj Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        return df

    def process_dataframe(self, df, n_steps):
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

    def scale_dataframe(self, df):
        """
        Scale dataframe
        Inputs: df
        Returns: df_scaled
        """
        feature_cols = df.columns[1:]  # All columns except the target
        target_col = df.columns[0]  # The first column is the target
        df_features_scaled = self.scaler_features.fit_transform(df[feature_cols])
        df_target_scaled = self.scaler_target.fit_transform(df[[target_col]])
        df_scaled = np.hstack([df_target_scaled, df_features_scaled])
        return df_scaled

    def split_data(self, scaled_data):
        """
        Split data into X and y
        Inputs: scaled_data
        Returns: X, y
        """
        X = scaled_data[:, 1:]  # All columns except the first one (target)
        y = scaled_data[:, 0]  # First column is the target
        return X, y

    def split_data_into_train_test(self, X, y, split_ratio=0.95):
        """
        Split data into train and test datasets
        Inputs: X, y, split_ratio
        Returns: X_train, X_test, y_train, y_test
        """
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        return X_train, X_test, y_train, y_test

    def get_guide(self, model):
        """
        Get guide
        Inputs: model
        Returns: guide
        """
        guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
        return guide

    def train(self, model, guide, num_iterations=1000, lr=0.01):
        """
        Train model
        Inputs: model, guide, num_iterations, lr
        Returns: svi
        """
        pyro.clear_param_store()
        svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
        for epoch in range(num_iterations):
            loss = svi.step(self.X_train_tensor, self.y_train_tensor)
            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")
        return svi

    def predict(self, model, guide, X):
        """
        Predict
        Inputs: model, guide, X
        Returns: y_pred_mean, y_pred_std
        """
        predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000, return_sites=("obs", "_RETURN"))
        samples = predictive(X)
        y_pred_samples = samples["_RETURN"].detach().numpy()
        y_pred_mean = y_pred_samples.mean(axis=0)
        y_pred_std = y_pred_samples.std(axis=0)
        return y_pred_mean, y_pred_std

    def plot_results(self, y_test, y_pred_mean, y_pred_std):
        """
        Plot results
        Inputs: y_test, y_pred_mean, y_pred_std
        Returns: None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='True Values', color='blue')
        plt.plot(y_pred_mean, label='Predictions', color='red')

        lower_bound = y_pred_mean - 2 * y_pred_std
        upper_bound = y_pred_mean + 2 * y_pred_std
        plt.fill_between(range(len(y_pred_mean)), lower_bound, upper_bound, color='red', alpha=0.3)
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def predict_next_day_close(self, model, guide):
        """
        Predict adj close for the next trading day
        Inputs: model, guide
        Returns: next_day_price
        """
        last_n_days = self.df['Adj Close'].values[-self.n_steps:]
        last_n_days_scaled = self.scaler_features.transform(last_n_days.reshape(1, -1))
        last_n_days_tensor = torch.tensor(last_n_days_scaled, dtype=torch.float32)
        next_day_mean, next_day_std = self.predict(model, guide, last_n_days_tensor)
        next_day_price = self.scaler_target.inverse_transform(next_day_mean.reshape(-1, 1)).flatten()[0]
        return next_day_price

    def predict_next_week_close(self, model, guide):
        """
        Predict adj close for the next week
        Inputs: model, guide
        Returns: next_week_predictions
        """
        last_n_days = self.df['Adj Close'].values[-self.n_steps:]
        next_week_predictions = []

        for _ in range(7):
            last_n_days_scaled = self.scaler_features.transform(last_n_days.reshape(1, -1))
            last_n_days_tensor = torch.tensor(last_n_days_scaled, dtype=torch.float32)
            next_day_mean, _ = self.predict(model, guide, last_n_days_tensor)
            next_day_price = self.scaler_target.inverse_transform(next_day_mean.reshape(-1, 1)).flatten()[0]
            next_week_predictions.append(next_day_price)
            last_n_days = np.append(last_n_days[1:], next_day_price)

        return next_week_predictions

def objective(trial, stock_predictor):
    """
    Optuna hyperparamter tuning
    Inputs: trial, stock_predictor
    Returns: rmse.item()
    """
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


# Uncomment for testing purposes
# stock_predictor = StockPredictorBNN('Stock Data/AAPL.csv')

# # Create an Optuna study and optimize the objective function
# study = optuna.create_study(direction='minimize')
# study.optimize(stock_predictor.objective, n_trials=50)

# print('Best trial:')
# trial = study.best_trial
# print(f'  RMSE: {trial.value}')
# print('  Best hyperparameters:')
# for key, value in trial.params.items():
#     print(f'    {key}: {value}')

# # Train the model with the best hyperparameters
# best_hid_dim = trial.params['hid_dim']
# best_prior_scale = trial.params['prior_scale']
# best_lr = trial.params['lr']

# best_model = BNN(in_dim=7, hid_dim=best_hid_dim, prior_scale=best_prior_scale)
# best_guide = stock_predictor.get_guide(best_model)
# best_svi = stock_predictor.train(best_model, best_guide, num_iterations=1000, lr=best_lr)

# # Predict on the test set and plot the results
# y_pred_mean, y_pred_std = stock_predictor.predict(best_model, best_guide, stock_predictor.X_test_tensor)
# stock_predictor.plot_results(stock_predictor.y_test, y_pred_mean, y_pred_std)

# # Predict the adjusted closing price
# next_day_price = stock_predictor.predict_next_day_close(best_model, best_guide)
# print(f'Predicted next day adjusted closing price: {next_day_price}')
# next_week_prices = stock_predictor.predict_next_week_close(best_model, best_guide)
# print(f'Predicted next week adjusted closing prices: {next_week_prices}')
