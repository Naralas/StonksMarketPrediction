import pandas as pd
import re
import numpy as np
import math

from itertools import tee, islice

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
            # get substring starting from what is after the dot 
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

def compute_RSI(df, n, price_column='Close', diff_column='Difference'):
    """ Code adapted from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas"""
    difference = df[diff_column] 

    # Make the positive gains (up) and negative gains (down) Series
    up, down = difference.copy(), difference.copy()
    up[up < 0] = 0
    down[down > 0] = 0


    # Calculate the SMA
    roll_up2 = up.rolling(n).mean()
    roll_down2 = down.abs().rolling(n).mean()

    # Calculate the RSI based on SMA
    rs = roll_up2 / roll_down2
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi

def get_data(path, file=None):
    if file is None:
        df = pd.read_csv(path, delimiter=' ')
    else:
        df = pd.read_csv(f"{path}/{file}", delimiter=' ')
    __trim_columns__(df)
    return df



if __name__ == '__main__':
    df = get_data('./data', 'AAPL.txt')
    df = get_data('./data/AAPL.txt')
    df['Difference'] =  compute_column_difference(df, column="Close")
    df['Diff4'] = compute_column_difference(df, column='Close', periods_offset=4)
    df['DiffPercent'] = compute_percentage_diff(df)
    df['Tendency'] = compute_tendency_percentage(df, thresh_diff=2.0)
    df['RSI'] = compute_RSI(df, 10)
    df['gap'] = compute_GAP(df)
    #print(df.head(20))

     