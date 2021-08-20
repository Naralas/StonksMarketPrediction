import pandas as pd
import re
import numpy as np
import math
import json

from pathlib import Path
from itertools import tee, islice
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, train_test_split

"""Helper functions file.

Contains functions for metrics, saving results, building datasets, etc.

Functions : 
    *save_predictions_heatmaps(path: string, metrics_dict: dict, metrics_names_list: list[str], [reversed:boolean]) 
                                                                        - Build and save the metrics as heatmaps to files in path given
    *save_dict(dict_save: dict, path:string) 
                                                                        - Saves a given dict in JSON format to the given path.
    *read_dict(path:)                                                   - Reads a dict in JSON format at the given path, 
                                                                        - Returns : read python dictionary
    *merge_metric_dicts(dict_1: dict, dict_2: dict)                     - Merges two given dictionaries, will modify dict_1 
    *build_sequences(data: list, [seq_len:int])                         - Builds sequences of samples of given length
                                                                        - Returns : Numpy array of samples sequences

    *get_timeseries_splits(X:Numpy array, y: Numpy array, [val_size:float], [n_splits:int])
                                                                        - Build datasets according to sklearn timeseries samples split of the given features and target
                                                                        - Returns : Numpy array of tuples splits in format : 
                                                                                (X_train, X_val, X_test, y_train, y_val, y_test) if val > 0.0
                                                                                (X_train, X_test, y_train, y_test) else
    *compute_GAP(close_series: Pandas.Series, open_series: Pandas.Series)
                                                                        - Builds a pandas series of price GAP (close at t-1 - open at t) in absolute values
                                                                        - Returns : Pandas.Series with gap feature (can contain nulls)
    *compute_tendency(price_series: Pandas.Series, [percentage:boolean], [thresh_diff:float], [labels:list[str]])
                                                                        - Compute the stocks tendency (up, down, stationary), if thresh_diff not given will not add stationary class
                                                                        - Percentage flag will consider the threshold as a percentage of the price return 
                                                                        - Returns : Pandas.Series of tendency (can contain nulls)
    *trim_columns(df:Pandas.DataFrame)                                  - Removes stock prefix from the columns (AAPL.Close -> Close), will modify the dataframe directly

    *merge_datasets(*args:Pandas.DataFrames):                           - Merge the given Pandas.DataFrame object into the first given. Will modify the first one.
                                                                        - Returns : Pandas.DataFrame (first passed dataframe with appened next args)
"""

def save_predictions_heatmaps(path, metrics_dict, metrics_names_list, reversed=False):
    """Builds metrics heatmaps and saves them into given path, for each n-days prediction.
    Reversed flag can be given if the metrics are better when lower (such as for regression)
    Format of images saved will be : {path}{N days prediction}{Metric Name}.png
    Args:
        path (string): Path to store heatmaps. Will overwrite files with same names
        metrics_dict (dict): Metrics dictionary in format {n_days:{metric_label:{stock:value}}}
        metrics_names_list (list[str]): List of metrics labels
        reversed (bool, optional): Reverse the metrics colors of the heatmap, i.e. should be set to True when lower is better . Defaults to False.
    """
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
    """Dumps the dictionary in JSON format.

    Args:
        dict_save (dict): Dict to save
        path (string): Folder path and file name
    """
    with open(path, 'w') as f:
        json.dump(dict_save, f)

def read_dict(path):
    """Reads the JSON dictionary at the given path

    Args:
        path (string): File path to read from.

    Returns:
        dict: The read python dict.
    """
    with open(path, 'r') as f:
        read_dict = json.load(f)

    return read_dict

def merge_metric_dicts(dict_1, dict_2):
    """Merges the 2 given dictionaries

    Args:
        dict_1 (dict): Dict that will be modified to add dict_2 values
        dict_2 (dict): Source dict to add into dict_1 (unchanged)
    """
    for predict_n, quot_metrics in dict_1.items():
        for quot, clf_metrics in quot_metrics.items():
            # merge dictionaries
            clf_metrics = {**clf_metrics, **dict_2[predict_n][quot]}
            dict_1[predict_n][quot] = clf_metrics

def build_sequences(data, seq_len=5):
    """Build sequences with given size. Mainly used for LSTM models.

    Args:
        data (list): Input samples
        seq_len (int, optional): Sequence length, e.g. number of periods for stocks. Defaults to 5.

    Returns:
        Numpy array: Numpy array of samples with included samples of given sequence length.
    """
    n_sequences = len(data) - seq_len
    return np.array([data[i:i+seq_len] for i in range(n_sequences)])


def get_timeseries_splits(X, y, val_size=0.3, n_splits=5):
    """Builds cross validation splits using time series split from Scikit-Learn.

    Args:
        X (Numpy array): Input features.
        y (Numpy array): Input target.
        val_size (float, optional): Size of the fraction of data that will be used as validation set. Defaults to 0.3.
        n_splits (int, optional): Number of cross-validation splits. Defaults to 5.

    Returns:
        Numpy array [tuple]:  Numpy array of size n_splits, containing tuples of samples in the format :
                                (X_train, X_val, X_test, y_train, y_val, y_test) if val > 0.0
                                (X_train, X_test, y_train, y_test) else
    """
    if n_splits == 1:
        return train_test_split(X,y, test_size=val_size, shuffle=False)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    splits = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if val_size > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=False)
            splits.append((X_train, X_val, X_test, y_train, y_val, y_test))
        else:
            splits.append((X_train, X_test, y_train, y_test))
        
    return np.array(splits, dtype=object)

def compute_GAP(close_series, open_series):
    """Compute the Pandas Series gap of stocks. The gap is defined as the price difference from the previous period to the current one, i.e. Close_t-1 - Open_t

    Args:
        close_series (Pandas.Series): Price closing series (unshifted).
        open_series (Pandas.Series): Price opening series (unshifted).

    Returns:
        Pandas.Series: Pandas series of prices difference, as absolute values.
    """
    shifted_close = close_series.shift(periods=1)
    # compute the difference between the pair of prices : yesterday's close - today's open
    gaps = list(map(lambda price_pair: price_pair[0] - price_pair[1], zip(shifted_close, open_series)))
    return np.abs(gaps)

def compute_tendency(price_series, percentage=False, thresh_diff=None, labels=['lower', 'stay', 'higher']):
    """Compute the Pandas Series of price tendencies, e.g. going up or down.
    If a threshold is given will add a stationary class for small upwards or downwards movements.
    Args:
        price_series (Pandas.Series): Price series, usually closing price-
        percentage (bool, optional): Consider price threshold as a percentage of the previous price. Defaults to False.
        thresh_diff ([float], optional): Price difference threshold for stationary class. Defaults to None.
        labels (list, optional): List of labels for the trends. Defaults to ['lower', 'stay', 'higher'].

    Raises:
        Exception: Threshold given but more or less than 3 labels were given.
        Exception: No threshold given but more or less than 2 labels were given.

    Returns:
        Pandas.Series: Trends series as strings, can contain nulls.
    """
    if percentage:
        price_series = price_series.pct_change()
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
    """Remove the prefix in the stocks DataFrame columns. E.g. "AAPL.Close -> Close"
    Standardize names.
    Args:
        df (Pandas.DataFrame): Dataframe with modified column names.
    """
    column_names = []
    for column in df.columns.values.tolist():
        # get substring starting from what is after the dot IE removing "AAPL" from "AAPL.Close"
        column_names.append(re.sub(r'^.*?\.', '', column))
    df.columns = column_names


def merge_datasets(*args):
    """Merge pandas dataframe passed as arguments.
    Will modify the first one passed.
    Raises:
        Exception: If the arguments contain other objects than Pandas DataFrames.
        Exception: If passed less than 2 dataframes.

    Returns:
        Pandas.DataFrame: First pandas dataframe passed with other dataframes appened to it.
    """
    if any((type(arg)) is not pd.DataFrame for arg in args):
        raise Exception("Arguments given to the function should be dataframes.") 
    if len(args) < 2:
        raise Exception("You need to pass multiple dataframes.")
    
    df_base = args[0]

    for df in args[1:]:
        df_base = df_base.append(df)
    
    return df_base



