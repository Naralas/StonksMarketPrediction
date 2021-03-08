import pandas as pd
import re

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
                bins=[min(df[diff_column]),0-thresh_diff, 0+thresh_diff, max(df[diff_column])],
                labels=labels)

    return tendencies



def get_data(folder_prefix, file):
    df = pd.read_csv(f"{folder_prefix}/{file}", delimiter=' ')
    __trim_columns__(df)
    return df





if __name__ == '__main__':
    df = get_data('./data', 'AAPL.txt')
    df['Difference'] = compute_price_difference(df)
    print(df.head())

     