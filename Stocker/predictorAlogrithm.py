"""
STOCKS = AAPL, MSFT, SPOT, TSLA

"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta, datetime
from xgboost import XGBRegressor, XGBClassifier
import plotly as py 
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
import requests, ta, math
from tqdm import tqdm_notebook
from fastai.tabular.core import add_datepart
import chart_studio.plotly as py
import cv

TRAIN_SIZE = 252*3  # 3 Years of data
VAL_SIZE = 252  # 1 Year worth of data
TRAIN_VAL_SIZE = TRAIN_SIZE + VAL_SIZE
H = 21  # AVG Number of Trading Days in a Month
N = 3  # Number of Lags
N_ESTIMATORS = 100  # Number of estimators
MAX_DEPTH = 3  # Max Tree depth 
LEARNING_RATE = 0.1  # Boosting learning rate
MIN_CHILD_WEIGHT = 1  # Minimum sum of instance weight
SUBSAMPLE = 1  # Subsample ratio of training instance
COLSAMPLE_BYTREE = 1  # Subsample ratio of coulumns when contructing each tree
COLSAMPLE_BYLEVEL = 1  # Subsample ratio of columns for each split in each level
GAMMA = 0  # Minimum loss reduction required to make further partition on a leaf node of the tree
MODEL_SEED = 100
FONTSIZE = 14
TICKLABELSIZE = 14

py.sign_in('cpong4', '2r45GCJHNyshuyUWeQbq')

def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_mae(a, b):
    """
    Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
    Returns a vector of len = len(a) = len(b)
    """
    return np.mean(abs(np.array(a)-np.array(b)))

def get_rmse(a, b):
    """
    Comp RMSE. a and b can be lists.
    Returns a scalar.
    """
    return math.sqrt(np.mean((np.array(a)-np.array(b))**2))

def get_mov_avg_std(df, col, N):
    """
    Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe. Can be of any length.
        col        : name of the column you want to calculate mean and std dev
        N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
    Outputs
        df_out     : same as df but with additional column containing mean and std dev
    """
    mean_list = df[col].rolling(window = N, min_periods=1).mean() # len(mean_list) = len(df)
    std_list = df[col].rolling(window = N, min_periods=1).std()   # first value will be NaN, because normalized by N-1
    
    # Add one timestep to the predictions
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))
    
    # Append mean_list to df
    df_out = df.copy()
    df_out[col + '_mean'] = mean_list
    df_out[col + '_std'] = std_list
    
    return df_out

def scale_row(row, feat_mean, feat_std):
    """
    Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
    Inputs
        row      : pandas series. Need to scale this.
        feat_mean: mean  
        feat_std : standard deviation
    Outputs
        row_scaled : pandas series with same length as row, but scaled
    """
    # If feat_std = 0 (this happens if adj_close doesn't change over N days), 
    # set it to a small number to avoid division by zero
    feat_std = 0.001 if feat_std == 0 else feat_std
    
    row_scaled = (row-feat_mean) / feat_std
    
    return row_scaled

def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_pred_eval_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test, col_mean, col_std, seed=100, n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=1, subsample=1, colsample_bytree=1, colsample_bylevel=1, gamma=0):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use XGBoost here.
    Inputs
        X_train_scaled     : features for training. Scaled to have mean 0 and variance 1
        y_train_scaled     : target for training. Scaled to have mean 0 and variance 1
        X_test_scaled      : features for test. Each sample is scaled to mean 0 and variance 1
        y_test             : target for test. Actual values, not scaled.
        col_mean           : means used to scale each sample of X_test_scaled. Same length as X_test_scaled and y_test
        col_std            : standard deviations used to scale each sample of X_test_scaled. Same length as X_test_scaled and y_test
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              : 
    Outputs
        rmse               : root mean square error of y_test and est
        mape               : mean absolute percentage error of y_test and est
        est                : predicted values. Same length as y_test
    '''

    model = XGBRegressor(seed=MODEL_SEED, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE, min_child_weight=MIN_CHILD_WEIGHT, subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE_BYTREE, colsample_bylevel=COLSAMPLE_BYLEVEL, gamma=GAMMA)
    # Train the model
    model.fit(X_train_scaled, y_train_scaled)
    
    # Get predicted labels and scale back to original range
    est_scaled = model.predict(X_test_scaled)
    est = est_scaled * col_std + col_mean

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test, est))
    mape = get_mape(y_test, est)
    
    return rmse, mape, est

def pred_xgboost(model, series, N, H):
    """
    Do recursive forecasting using xgboost
    Inputs
        model : the xgboost model
        series: numpy array of shape (len(series),). The time series we want to do recursive forecasting on
        N     : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H     : forecast horizon
    Outputs
        Times series of predictions. Numpy array of shape (H,).
    """
    forecast = series.copy()
    for n in range(H):
        est = model.predict(forecast[-N:].reshape(1,-1))
        forecast = np.append(forecast, est)

    return forecast[-H:]

def getMeanofAdjClose(df):
    merging_keys = ['order_day']
    lag_cols = ['adj_close']

    # Get mean of adj_close of each month
    df_gb = df.groupby(['month'], as_index=False).agg({'adj_close':'mean'})
    df_gb = df_gb.rename(columns={'adj_close':'adj_close_mean'})

    # Merge to main df
    df = df.merge(df_gb, left_on=['month'], right_on=['month'],how='left').fillna(0)

    # Merge to main df
    shift_range = [x+1 for x in range(2)]

    for shift in tqdm_notebook(shift_range):
        train_shift = df[merging_keys + lag_cols].copy()
        
        # E.g. order_day of 0 becomes 1, for shift = 1.
        # So when this is merged with order_day of 1 in df, this will represent lag of 1.
        train_shift['order_day'] = train_shift['order_day'] + shift
        
        foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
        train_shift = train_shift.rename(columns=foo)

        df = pd.merge(df, train_shift, on=merging_keys, how='left') #.fillna(0)
        
    del train_shift

def getMeanVolume(df):
    # Get mean of volume of each month
    df_gb = df.groupby(['month'], as_index=False).agg({'volume':'mean'})
    df_gb = df_gb.rename(columns={'volume':'volume_mean'})

    # Merge to main df
    df = df.merge(df_gb, left_on=['month'], right_on=['month'],how='left').fillna(0)


def processDataFromFile(csv):
    df = pd.read_csv(csv, sep=',', parse_dates=['Date'])
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')  # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]  # Make columns lowercase, remove spacing between words
    df['month'] = df['date'].dt.month  # Get month of each sample
    df.sort_values(by='date', inplace=True, ascending=True)  # Sort by datetime
    getMeanVolume(df)
    df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)  # Cannot use these columns as features
    df['order_day'] = [x for x in list(range(len(df)))]
    merging_keys = ['order_day']
    lag_cols = ['adj_close']
    shift_range = [x+1 for x in range(N)]

    for shift in tqdm_notebook(shift_range):
        train_shift = df[merging_keys + lag_cols].copy()
        # E.g. order_day of 0 becomes 1, for shift = 1.
        # So when this is merged with order_day of 1 in df, this will represent lag of 1.
        train_shift['order_day'] = train_shift['order_day'] + shift
        foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
        train_shift = train_shift.rename(columns=foo)
        df = pd.merge(df, train_shift, on=merging_keys, how='left').fillna(0)
        
    del train_shift    
    df = df[N:]  # Remove the first N rows which contain NaNs
    getMeanofAdjClose(df)
    cols_list = ["adj_close"]

    for col in cols_list:
        df = get_mov_avg_std(df, col, N)
        
    i = 1008
    train, test = splitDataIntoTrainingTesting(df, i)
    train_scaled, test_scaled, scaler = scaleTrainTest(train, test, cols_list)

    X_train, y_train, x_train_scaled, y_train_scaled = splitTrainTestIntoXY(train, test, train_scaled, test_scaled)
    train_model(X_train, y_train, x_train_scaled, y_train_scaled, scaler)

def splitDataIntoTrainingTesting(df, i):
    train = df[i-TRAIN_VAL_SIZE:i]
    test = df[i:i+H]
    return train, test

def scaleTrainTest(train, test, cols_list):
    cols_to_scale = ["adj_close"]

    for n in range(1,N+1):
        cols_to_scale.append("adj_close_lag_"+str(n))
        
    # Do scaling for train set
    # Here we only scale the train dataset, and not the entire dataset to prevent information leak
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[cols_to_scale])

    # Convert the numpy array back into pandas dataframe
    train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
    train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]

    # Do scaling for test set
    test_scaled = test[['date']]
    for col in tqdm_notebook(cols_list):
        feat_list = [col + '_lag_' + str(shift) for shift in range(1, N+1)]
        temp = test.apply(lambda row: scale_row(row[feat_list], row[col+'_mean'], row[col+'_std']), axis=1)
        test_scaled = pd.concat([test_scaled, temp], axis=1)
        
    # Now the entire test set is scaled
    return train, test, scaler

def splitTrainTestIntoXY(train, test, train_scaled, test_scaled):
    features = []
    for n in range(1,N+1):
        features.append("adj_close_lag_"+str(n))
    target = "adj_close"

    # Split into X and y
    X_train = train[features]
    y_train = train[target]
    X_sample = test[features]
    y_sample = test[target]

    # Split into X and y
    X_train_scaled = train_scaled[features]
    y_train_scaled = train_scaled[target]
    X_sample_scaled = test_scaled[features]
    return X_train, y_train, X_train_scaled, y_train_scaled

def train_model(X_train, y_train, X_train_scaled, y_train_scaled, scaler):
    # Create the model
    # model = XGBRegressor(objective ='reg:squarederror', seed=MODEL_SEED, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE, min_child_weight=MIN_CHILD_WEIGHT, subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE_BYTREE, colsample_bylevel=COLSAMPLE_BYLEVEL, gamma=GAMMA)
    model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=100, silent=None, subsample=1, verbosity=1)
    # Train the regressor
    model.fit(X_train_scaled.to_numpy(), y_train_scaled.to_numpy())
    # Do prediction on train set
    est_scaled = model.predict(X_train_scaled.to_numpy())
    est = est_scaled * math.sqrt(scaler.var_[0]) + scaler.mean_[0]

    # Calculate RMSE
    print("RMSE on train set = %0.3f" % math.sqrt(mean_squared_error(y_train, est)))

    # Calculate MAPE
    print("MAPE on train set = %0.3f%%" % get_mape(y_train, est))
    

def main():
    processDataFromFile('Stock Data/AAPL.csv')
    # AAPL_FEATURES, AAPL_LABELS = processDataFromFile('Stock Data/HistoricalData_AAPL.csv')
    # GOOGL_FEATURES, GOOGL_LABELS = processDataFromFile('Stock Data/HistoricalData_GOOGL.csv')
    # MSFT_FEATURES, MSFT_LABELS = processDataFromFile('Stock Data/HistoricalData_MSFT.csv')
    # VOO_FEATURES, VOO_LABELS = processDataFromFile('Stock Data/HistoricalData_VOO.csv')

    #ml_algorithm(AAPL_FEATURES, AAPL_LABELS)
    #ml_algorithm(GOOGL_FEATURES, GOOGL_LABELS)

    # sentiment_analysis('AAPL')


main()