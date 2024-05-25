import pyro
import pyro.distributions as dist
import pyro.nn as pyro_nn
import torch.nn as nn
from pyro.nn import PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import torch
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pyro.infer import Predictive


class BNN(pyro_nn.PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=5, prior_scale=10):
        super().__init__()

        self.activation = nn.Tanh()
        self.layer1 = pyro_nn.PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = pyro_nn.PyroModule[nn.Linear](hid_dim, out_dim)

        self.layer1.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(0., prior_scale).expand([out_dim, hid_dim]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(0., prior_scale).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(1., 1.))

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu


def read_data_from_file(csv):
    df = pd.read_csv(csv, usecols=['Date', 'Adj Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df


def process_dataframe(df, n_steps):
    df = dc(df)
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Adj Close'].shift(i)
    df.dropna(inplace=True)
    return df


def scale_dataframe(shifted_df):
    shifted_df_as_np = shifted_df.to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    return shifted_df_as_np, scaler


def split_data(shifted_df_as_np):
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    return X, y


def split_data_into_train_test(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test


def plot_results(y_test, y_pred_samples):
    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_pred_std = np.std(y_pred_samples, axis=0)

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


def generate_prediction_samples(model, guide, X_test, num_samples=1000):
    predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("obs",))
    samples = predictive(X_test)
    y_pred_samples = samples["obs"].detach().numpy()
    return y_pred_samples


def guide(bnn, x, y=None):
    fc1w_mu = pyro.param("fc1w_mu", torch.randn_like(bnn.layer1.weight))
    fc1w_sigma = pyro.param("fc1w_sigma", torch.ones_like(bnn.layer1.weight))
    fc1b_mu = pyro.param("fc1b_mu", torch.randn_like(bnn.layer1.bias))
    fc1b_sigma = pyro.param("fc1b_sigma", torch.ones_like(bnn.layer1.bias))
    fc2w_mu = pyro.param("fc2w_mu", torch.randn_like(bnn.layer2.weight))
    fc2w_sigma = pyro.param("fc2w_sigma", torch.ones_like(bnn.layer2.weight))
    fc2b_mu = pyro.param("fc2b_mu", torch.randn_like(bnn.layer2.bias))
    fc2b_sigma = pyro.param("fc2b_sigma", torch.ones_like(bnn.layer2.bias))
    priors = {
        'layer1.weight': dist.Normal(fc1w_mu, fc1w_sigma).to_event(2),
        'layer1.bias': dist.Normal(fc1b_mu, fc1b_sigma).to_event(1),
        'layer2.weight': dist.Normal(fc2w_mu, fc2w_sigma).to_event(2),
        'layer2.bias': dist.Normal(fc2b_mu, fc2b_sigma).to_event(1),
    }
    lifted_module = pyro.random_module("module", bnn, priors)
    return lifted_module()


def train_bnn(bnn, guide, X_train, y_train, num_iterations=1000, learning_rate=0.01):
    pyro.clear_param_store()
    optimizer = Adam({"lr": learning_rate})
    svi = SVI(bnn, guide, optimizer, loss=Trace_ELBO())

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    for i in range(num_iterations):
        loss = svi.step(X_train, y_train)
        if i % 100 == 0:
            print(f'Iteration {i} - Loss: {loss}')


def predict_next_day(model, guide, X_last, scaler_target):
    X_last = torch.tensor(X_last, dtype=torch.float32)
    predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=("obs",))
    samples = predictive(X_last)
    predicted_price = samples["obs"].mean(0).item()
    return scaler_target.inverse_transform([[predicted_price]])[0][0]


def predict_next_week(model, guide, df, scaler_features, scaler_target, n_steps):
    last_n_days = df['Adj Close'].values[-n_steps:]
    next_week_predictions = []

    for _ in range(7):
        last_n_days_df = pd.DataFrame([last_n_days], columns=[f'Close(t-{i})' for i in range(n_steps, 0, -1)])
        last_n_days_scaled = scaler_features.transform(last_n_days_df)
        next_day_price = predict_next_day(model, guide, last_n_days_scaled, scaler_target)
        next_week_predictions.append(next_day_price)
        last_n_days = np.append(last_n_days[1:], next_day_price)
    return next_week_predictions


def objective(trial, X_train, y_train, X_val, y_val, input_dim, output_dim):
    hid_dim = trial.suggest_int('hid_dim', 5, 50)
    prior_scale = trial.suggest_float('prior_scale', 1, 20)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    num_iterations = trial.suggest_int('num_iterations', 500, 2000)

    bnn = BNN(in_dim=input_dim, out_dim=output_dim, hid_dim=hid_dim, prior_scale=prior_scale)
    train_bnn(bnn, lambda x, y: guide(bnn, x, y), X_train, y_train, num_iterations=num_iterations, learning_rate=learning_rate)

    predictive = Predictive(bnn, guide=lambda x, y: guide(bnn, x, y), num_samples=1000, return_sites=("obs",))
    samples = predictive(torch.tensor(X_val, dtype=torch.float32))
    y_pred = samples["obs"].mean(0).detach().numpy()
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return rmse


def main():
    df = read_data_from_file('Stock Data/AAPL.csv')
    processed_df = process_dataframe(df, N_STEPS)
    scaled_data, scaler = scale_dataframe(processed_df)
    X, y = split_data(scaled_data)
    X_train, X_test, y_train, y_test = split_data_into_train_test(X, y)
    split_index = int(len(X_train) * 0.8)
    X_train, X_val, y_train, y_val = X_train[:split_index], X_train[split_index:], y_train[:split_index], y_train[split_index:]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, X.shape[1], 1), n_trials=50)
    print('Best hyperparameters:', study.best_params)
    print('Best RMSE:', study.best_value)
    best_params = study.best_params
    bnn = BNN(in_dim=X.shape[1], out_dim=1, hid_dim=best_params['hid_dim'], prior_scale=best_params['prior_scale'])
    train_bnn(bnn, lambda x, y: guide(bnn, x, y), X_train, y_train, num_iterations=best_params['num_iterations'], learning_rate=best_params['learning_rate'])

    # Make predictions
    next_day_price = predict_next_day(bnn, lambda x, y: guide(bnn, x, y), X_test[0], scaler)
    print(f'Predicted next day adjusted close price: {next_day_price}')
    next_week_prices = predict_next_week(bnn, lambda x, y: guide(bnn, x, y), df, scaler, scaler, N_STEPS)
    print(f'Predicted next 7 days adjusted close prices: {next_week_prices}')

if __name__ == "__main__":
    N_STEPS = 7
    main()
    

if __name__ == "__main__":
    main()
