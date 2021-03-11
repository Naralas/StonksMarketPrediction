import pandas as pd
import re
import numpy as np
import math

from itertools import tee

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def __trim_columns__(df):
        column_names = []
        for column in df.columns.values.tolist():
            # get substring starting from what is after the dot 
            column_names.append(re.sub(r'^.*?\.', '', column))
        df.columns = column_names

def compute_price_difference(df, price_column='Close'):
    values = df[price_column]
    differences = list(map(lambda pair: pair[1]-pair[0], pairwise(values)))
    differences.insert(0, 0)

    return differences

def compute_tendency(df, diff_column='Difference', thresh_diff=0.04, labels=['lower', 'stay', 'higher']):
    tendencies = pd.cut(x=df[diff_column], 
                bins=[-math.inf,0-thresh_diff, 0+thresh_diff, math.inf],
                labels=labels)

    return tendencies

def compute_percentage_diff(df, price_column='Close', diff_column='Difference'):
    return df.apply(lambda r: r[diff_column] / (r[price_column] - r[diff_column]) * 100.0, axis=1)
    
def compute_tendency_percentage(df, price_column='Close', diff_column='Difference', thresh_diff=1.0, labels=['lower', 'stay', 'higher']):
    percentages = compute_percentage_diff(df, price_column, diff_column)
    tendencies = pd.cut(x=percentages,
                bins=[-math.inf, 0-thresh_diff, 0+thresh_diff, math.inf],
                labels=labels)

    return tendencies

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



def get_data(folder_prefix, file):
    df = pd.read_csv(f"{folder_prefix}/{file}", delimiter=' ')
    __trim_columns__(df)
    return df





if __name__ == '__main__':
    df = get_data('./data', 'AAPL.txt')
    df['Difference'] = compute_price_difference(df)
    df['DiffPercent'] = compute_percentage_diff(df)
    df['Tendency'] = compute_tendency_percentage(df, thresh_diff=2.0)
    df['RSI'] = compute_RSI(df, 10)
    print(df.head(20))

     