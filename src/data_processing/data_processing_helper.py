import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config.variables import day_split_test, TRAIN_SIZE_RATIO


def preprocess_data(ts_dt):
    """
    Convert type of 'Date_time' column to datetime64 
    """
    ts_dt['Date_time'] = pd.to_datetime(ts_dt['Date_time'])
    return ts_dt

def scale_data(df):
    df_copy = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    TARGET = 'totalU'
    FEATURES = [feature for feature in df.columns if feature != TARGET]

    X_scaler = scaler.fit_transform(df_copy[FEATURES])
    y_scaler = scaler.fit_transform(np.asarray(df_copy[TARGET]).reshape(-1, 1))

    return scaler, X_scaler, y_scaler


def split_data(dataframe, X_scaler, y_scaler):
    """
    This function splits the input dataset into training, validation, and test sets
    """
    dates = dataframe.index
    boundary_idx = dataframe.index.searchsorted(pd.Timestamp(day_split_test))

    train_val_dates, train_val_X, train_val_y = dates[:boundary_idx], X_scaler[:boundary_idx], y_scaler[:boundary_idx]
    _, X_test, y_test = dates[boundary_idx:], X_scaler[boundary_idx:], y_scaler[boundary_idx:]

    train_size = int(len(train_val_X) * TRAIN_SIZE_RATIO)

    _, X_train, y_train = train_val_dates[:train_size], train_val_X[:train_size, :], \
        train_val_y[:train_size, :]
    _, X_val, y_val = train_val_dates[train_size:], train_val_X[train_size:], \
        train_val_y[train_size:, :]

    return X_train, y_train, X_val, y_val, X_test, y_test
