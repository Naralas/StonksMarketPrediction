import pandas as pd
import re
import numpy as np
import math
import ta

from pathlib import Path
from itertools import tee, islice
from sklearn import preprocessing

from sklearn.model_selection import TimeSeriesSplit, train_test_split


def build_sequences(data, seq_len=5):
    n_sequences = len(data) - seq_len
    return np.array([data[i:i+seq_len] for i in range(n_sequences)])


def get_timeseries_splits(X, y, val_size=0.3, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    splits = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if val_size != 0.0:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=False)
            splits.append((X_train, X_val, X_test, y_train, y_val, y_test))
        else:
            splits.append((X_train, X_test, y_train, y_test))
        
    return np.array(splits, dtype=object)

def compute_GAP(close_series, open_series):
    shifted_close = close_series.shift(periods=1)
    # compute the difference between the pair of prices : yesterday's close - today's open
    gaps = list(map(lambda price_pair: price_pair[0] - price_pair[1], zip(shifted_close, open_series)))
    return np.abs(gaps)

def compute_tendency(series, percentage=False, thresh_diff=None, labels=['lower', 'stay', 'higher']):
    if percentage:
        series = series.pct_change()
    if thresh_diff is not None:
        if len(labels) is not 3:
            raise Exception(f"If the threshold is specified, you have to specify 3 labels, given : {labels}")
        else:
            return pd.cut(series,bins=[-math.inf, 0-thresh_diff, 0+thresh_diff, math.inf],labels=labels)
    else:
        if len(labels) is not 2:
            raise Exception(f"If no threshold is specified, you have to provide 2 labels, given : {labels}")
        else:
            return pd.cut(series,bins=[-math.inf, 0, math.inf],labels=labels)

def trim_columns(df):
    column_names = []
    for column in df.columns.values.tolist():
        # get substring starting from what is after the dot IE removing "AAPL" from "AAPL.Close"
        column_names.append(re.sub(r'^.*?\.', '', column))
    df.columns = column_names


def merge_datasets(*args):
    if any((type(arg)) is not pd.DataFrame for arg in args):
        raise Exception("Arguments given to the function should be dataframes.") 
    if len(args) < 2:
        raise Exception("You need to pass multiple dataframes.")
    
    df_base = args[0]

    for df in args[1:]:
        df_base = df_base.append(df)
    
    return df_base

