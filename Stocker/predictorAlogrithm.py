"""
STOCKS = AAPL, MSFT, SPOT, TSLA

"""


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

if __name__ == "__main__":
    main()
