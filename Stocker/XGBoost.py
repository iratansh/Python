"""
STOCKS = AAPL, MSFT, SPOT, TSLA
"""

from datetime import date, timedelta, datetime
from xgboost import XGBRegressor, XGBClassifier
import plotly.graph_objects as go
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dc
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
        

def readDataFromFile(csv):
    df = pd.read_csv(csv, usecols=['Date', 'Close'])
    return df

def process_dataframe(df, n_steps):
    df = dc(df)
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

def scale_dataframe(shifted_df):
    shifted_df_as_np = shifted_df.to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    return shifted_df_as_np

def split_data(shifted_df_as_np):
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    return X, y

def split_data_into_train_test(X, y, split_index):
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

def reshape_datasets(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape((-1, 7, 1))
    X_test = X_test.reshape((-1, 7, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    return X_train, X_test, y_train, y_test

def convert_to_tensors(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    return X_train, X_test, y_train, y_test

def loaders(train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def train_one_epoch(model, train_loader, epoch):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(DEVICE), batch[1].to(DEVICE)
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1, avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch(model, test_loader):
    model.train(False)
    running_loss = 0.0
    loss_function = nn.MSELoss()

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(DEVICE)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    
    avg_loss_across_batches = running_loss / len(test_loader)
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print()

def main():
    df = readDataFromFile('Stock Data/AAPL.csv')
    shifted_df = process_dataframe(df, 7)
    shifted_df_as_np = scale_dataframe(shifted_df)
    X, y = split_data(shifted_df_as_np)
    X = dc(np.flip(X, axis=1))
    split_index = int(len(X) * 0.95)
    X_train, X_test, y_train, y_test = split_data_into_train_test(X, y, split_index)
    X_train, X_test, y_train, y_test = reshape_datasets(X_train, X_test, y_train, y_test )
    X_train, X_test, y_train, y_test = convert_to_tensors(X_train, X_test, y_train, y_test)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader, test_loader = loaders(train_dataset, test_dataset)


    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(DEVICE), batch[1].to(DEVICE)
        break
    model = LSTM(1, 4, 1)
    model.to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, train_loader, epoch)
        validate_one_epoch(model, test_loader)
    # print(model)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # print(train_dataset)

if __name__ == "__main__":
    main()
