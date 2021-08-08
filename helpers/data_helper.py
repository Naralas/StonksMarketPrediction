import pandas as pd
import re
import numpy as np
import math
import ta
import json

from pathlib import Path
from itertools import tee, islice
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, train_test_split

def save_predictions_heatmaps(path, metrics_dict, metrics_names_list, reversed=False):
    for predict_n, quot_metrics in metrics_dict.items():
        metrics_df = pd.DataFrame.from_dict(quot_metrics).T
    
        for metric in metrics_names_list:
            filtered_df = metrics_df.applymap(lambda metrics: metrics[metric])
            filtered_df = filtered_df.round(4)
            plt.figure()
            if reversed:
                heatmap = sns.heatmap(filtered_df, cmap ='mako_r', linewidths = 0.5, annot = True)
            else:
                heatmap = sns.heatmap(filtered_df, cmap ='mako', linewidths = 0.5, annot = True)
            heatmap.figure.savefig(fr"{path}{predict_n}_{metric}.png")
            plt.close()

def save_dict(dict_save, path):
    with open(path, 'w') as f:
        json.dump(dict_save, f)

def read_dict(path):
    with open(path, 'r') as f:
        read_dict = json.load(f)

    return read_dict

def merge_metric_dicts(dict_1, dict_2):
    for predict_n, quot_metrics in dict_1.items():
        for quot, clf_metrics in quot_metrics.items():
            # merge dictionaries
            clf_metrics = {**clf_metrics, **dict_2[predict_n][quot]}
            dict_1[predict_n][quot] = clf_metrics

def build_sequences(data, seq_len=5):
    n_sequences = len(data) - seq_len
    return np.array([data[i:i+seq_len] for i in range(n_sequences)])


def get_timeseries_splits(X, y, val_size=0.3, n_splits=5):
    if n_splits == 1:
        return train_test_split(X,y, test_size=val_size, shuffle=False)

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

def compute_tendency(price_series, percentage=False, thresh_diff=None, labels=['lower', 'stay', 'higher']):
    if percentage:
        #print(price_series)
        price_series = price_series.pct_change()
        #print(price_series)
    if thresh_diff is not None:
        if len(labels) is not 3:
            raise Exception(f"If the threshold is specified, you have to specify 3 labels, given : {labels}")
        else:
            return pd.cut(price_series, bins=[-math.inf, 0-thresh_diff, 0+thresh_diff, math.inf],labels=labels)
    else:
        if len(labels) is not 2:
            raise Exception(f"If no threshold is specified, you have to provide 2 labels, given : {labels}")
        else:
            return pd.cut(price_series, bins=[-math.inf, 0, math.inf],labels=labels)

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

