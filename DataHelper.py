import pandas as pd
import re
import numpy as np
import math
import ta

from pathlib import Path
from itertools import tee, islice
from sklearn import preprocessing


# modified from https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable, offset=None):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    if offset is None:
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    else:
        # get iterators for a and b with offseted by n (EG zipped, with 4 : [(0,4),(1,5),...])
        # islice returns iterator following : (iterable, start, stop, step)
        a = islice(iterable, 0, len(iterable), 1) 
        b = islice(iterable, offset, len(iterable), 1)
        return zip(a, b)

def __trim_columns__(df):
        column_names = []
        for column in df.columns.values.tolist():
            # get substring starting from what is after the dot IE removing "AAPL" from "AAPL.Close"
            column_names.append(re.sub(r'^.*?\.', '', column))
        df.columns = column_names

def shift_values(df, column, periods=-1):
    shifted_column = df[column]
    shifted_column = shifted_column.shift(periods)    
    return shifted_column

def compute_column_difference(df, column='Close', periods_offset=1):
    values = df[column]
    differences = list(map(lambda pair: pair[1]-pair[0], pairwise(values, offset=periods_offset)))
    for _ in range(periods_offset):
        differences.insert(0, np.NaN)

    return differences

def compute_tendency(df, diff_column='Difference', price_column='Close',
        thresh_diff=None, labels=['lower', 'stay', 'higher'], percentage=False):
    if percentage:
        return compute_tendency_percentage(df, diff_column=diff_column, price_column=price_column, thresh_diff=thresh_diff, labels=labels)
    else:
        return pd.cut(x=df[diff_column], 
                    bins=[-math.inf,0-thresh_diff, 0+thresh_diff, math.inf],
                    labels=labels)


def compute_percentage_diff(df, price_column='Close', diff_column='Difference'):
    return df.apply(lambda r: r[diff_column] / (r[price_column] - r[diff_column]) * 100.0, axis=1)
    
def compute_tendency_percentage(df, price_column='Close', diff_column='Difference', thresh_diff=None, labels=['lower', 'stay', 'higher']):
    percentages = compute_percentage_diff(df, price_column, diff_column)

    if thresh_diff is not None:
        tendencies = pd.cut(x=percentages,
                    bins=[-math.inf, 0-thresh_diff, 0+thresh_diff, math.inf],
                    labels=labels)
    else:
        tendencies = pd.cut(x=percentages, bins=[-math.inf, 0, math.inf], labels=labels)

    return tendencies

def compute_GAP(df, close_column='Close', open_column='Open'):
    shifted_close = shift_values(df, close_column, periods=1)
    # compute the difference between the pair of prices : yesterday's close - today's open
    gaps = list(map(lambda price_pair: price_pair[0] - price_pair[1], zip(shifted_close, df[open_column])))
    return gaps

def compute_RSI(df, n, price_column='Close'):
    return ta.momentum.RSIIndicator(df[price_column], window=n).rsi()

def compute_SMA(df, price_column, n=10):
    return ta.trend.SMAIndicator(df[price_column], window=n).sma_indicator()

def get_data(path, file=None):
    if file is None:
        df = pd.read_csv(path, delimiter=' ')
    else:
        df = pd.read_csv(f"{path}/{file}", delimiter=' ')
    __trim_columns__(df)
    return df

def import_folder(path, files_pattern="*", recursive=False, 
        compute_features=True, predict_n=1, normalize_features=False):

    files = Path(path)
    files = files.rglob(files_pattern) if recursive else files.glob(files_pattern)
    if not compute_features:
        return [(get_data(file), file.name) for file in files]
    else:
        return  [
                    (features_pipeline(file, predict_n=predict_n, normalize_features=normalize_features), file.name) 
                for file in files]
    

def merge_datasets(*args):
    if any((type(arg)) is not pd.DataFrame for arg in args):
        raise Exception("Arguments given to the function should be dataframes.") 
    if len(args) < 2:
        raise Exception("You need to pass multiple dataframes.")
    
    df_base = args[0]

    for df in args[1:]:
        df_base = df_base.append(df)
    
    return df_base


def normalize_df(df, features, scale=(-1,1)):
    # filter to get only the numerical columns
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    # filter to find the intersection with the features (that need to be scaled)
    target_column_names = [col for col in numerical_columns if col in features]

    # filter the columns and normalize
    df_n = df.loc[:, target_column_names]
    minmax_scaler = preprocessing.MinMaxScaler(scale)
    df_n = pd.DataFrame(minmax_scaler.fit_transform(df_n))
    df_n.columns = target_column_names


    # replace the columns in the dataframe with the normalized data
    df.loc[:, target_column_names] = df_n
    return df

def features_pipeline(path, price_column='Close', predict_n=1, thresh_diff=None, normalize_features=False, base_features_normalize=[], verbose=False):
    df = get_data(path)
    dataset_column_names = df.columns.values
    
    df['Difference'] = compute_column_difference(df, column=price_column, periods_offset=predict_n)
    df['PercentageDiff'] = compute_percentage_diff(df)
    
    if thresh_diff is None:
        df['Tendency'] = compute_tendency_percentage(df, diff_column='Difference', labels=['lower','higher'])
    else:
        df['Tendency'] = compute_tendency_percentage(df, diff_column='Difference', thresh_diff=thresh_diff)

    if verbose:
        value_counts = df.Tendency.value_counts().to_dict()
        for value, count in value_counts.items():
            print(f"[{value}] : {count} ({count * 100.0 / len(df['Tendency']):.1f}%)")
            
    df['SMA(10)'] = compute_SMA(df, price_column, n=10)
    df['SMA(20)'] = compute_SMA(df, price_column, n=20)
    df['EMA(14)'] = ta.trend.EMAIndicator(df[price_column], window=14).ema_indicator()
    df['EMA_Diff'] = compute_column_difference(df, column="EMA(14)", periods_offset=predict_n)
    df['SMA(20) - SMA(10)'] = compute_SMA(df, price_column, n=20) - compute_SMA(df, price_column, n=10)
    df['LowLen'] = df.apply(lambda r: np.minimum(r['Open'], r['Close']) - r['Low'], axis=1)
    df['RSI(14)'] = compute_RSI(df, n=14, price_column=price_column)
    df['GAP'] = compute_GAP(df)
    df['RSI_Diff'] = compute_column_difference(df, column='RSI(14)', periods_offset=predict_n)
    df['Volume_diff'] = compute_column_difference(df, column='Volume')
    df['Next'] = shift_values(df, column='Tendency', periods=-predict_n)
    macd = ta.trend.MACD(df[price_column])
    df['MACD'] = macd.macd()
    df['MACD_diff'] = macd.macd_diff()
    df['MACD_signal'] = macd.macd_signal()
    df['BodyLen'] = (df['Close'] - df['Open']).abs()
    bg_band = ta.volatility.BollingerBands(df[price_column])
    df['BG_L_Band'] = bg_band.bollinger_lband()
    df['BG_H_Band'] = bg_band.bollinger_hband()
    df['BG_L_Band_Indicator'] = bg_band.bollinger_lband_indicator()
    df['BG_H_Band_Indicator'] = bg_band.bollinger_hband_indicator()
    df['ROC'] = ta.momentum.ROCIndicator(df[price_column]).roc()
    df['StochOsc'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df[price_column]).stoch()
    
    df = df.dropna()

    #print(f"Before : {df.isna().any()}")
    feature_names = [col for col in df.columns.values if col not in dataset_column_names]

    if normalize_features:
        df = normalize_df(df, feature_names + base_features_normalize)
    #print(f"After : {df.isna().any()}")

    df = df.dropna()

    return df, feature_names



if __name__ == '__main__':
    df = get_data('./data', 'AAPL.txt')
    df = get_data('./data/AAPL.txt')
    df['Difference'] =  compute_column_difference(df, column="Close")
    df['Diff4'] = compute_column_difference(df, column='Close', periods_offset=4)
    df['DiffPercent'] = compute_percentage_diff(df)
    df['Tendency'] = compute_tendency_percentage(df, thresh_diff=2.0)
    df['RSI(14)'] = compute_RSI(df, 14)
    df['gap'] = compute_GAP(df)

    df, feature_names = features_pipeline('./data/AAPL.txt', 'Close', 1,  
        thresh_diff=None, normalize_features=True, base_features_normalize=['Volume'], verbose=False)

    print(df.head(20), feature_names)
    df_merged = merge_datasets(df, df, df, df)

    datasets = import_folder('./data', files_pattern="*.txt")
    print([x[1] for x in datasets])