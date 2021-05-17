import pandas as pd
import re
import numpy as np
import math
import ta

from pathlib import Path
from sklearn import preprocessing
from helpers.data_helper import *


default_features_list = ['Close', 'Volume', 'MACD_diff', 'RSI(14)', 'PercentageDiff', 'LowLen', 'RSI_diff']


class StocksDataWrapper:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, selector):
        return self.df[selector]

    def __setitem__(self, selector, item):
        self.df[selector] = item

    def __len__(self):
        return len(self.df)

    def head(self, n=5):
        return self.df.head(n)

    def get_df(self):
        return self.df

    def append_dataset(self, other):
        self.df = merge_datasets(self.df, other)

    def replace_pct_change(self, features_list):
        for col in features_list:
            df[col] = df[col].pct_change()


    def normalize_data(self, features_list=[], scale=(0,1), inplace=True):
        self.raw_df = self.df.copy()


        if len(features_list) == 0:
            features_list = self.get_numerical_features()

        self.scaled_features = features_list

        # filter the columns and normalize
        df_n = self.df.loc[:, features_list]
        self.minmax_scaler = preprocessing.MinMaxScaler(scale)


        df_n = pd.DataFrame(self.minmax_scaler.fit_transform(df_n))
        df_n.columns = features_list
    
        if inplace:
            # replace the columns in the dataframe with the normalized data
            self.df.loc[:, features_list] = df_n
        else:
            df_copy = self.df.copy()
            df_copy.loc[:, features_list] = df_n
            return df_copy
    
    
    def get_datasets(self, n_splits=5, val_size=0.3, sequences=False, seq_len=5, y_column='NextPrice',
                        features_list=default_features_list):

        dataset = self.df.copy()
        # filter only the features columns in the data and the target column
        dataset = dataset.loc[:, features_list + [y_column]]

        # replace the textual data (tendencies) by numerical values
        for col in dataset.columns:
            dataset[col] = dataset[col].replace({'higher':2, 'stay':1, 'lower':0})

        # remove the target column and build sequences if needed
        X = dataset.loc[:, dataset.columns != y_column].values
        if sequences:
            # X.shape = (X_a, X_b) -> (X_a, seq_len, X_b)
            X = build_sequences(X, seq_len=seq_len)

        y = dataset[y_column].values
        
        return get_timeseries_splits(X, y, val_size=val_size, n_splits=n_splits)



    def get_numerical_features(self):
        return self.df.select_dtypes(include=np.number).columns.tolist()
        
    def get_numerical_columns(self):
        return self.df.loc[:, self.get_numerical_columns()]
    
    def get_unscaled_values(self, values, base_name='Close'):
        df = pd.DataFrame(data={base_name:values})
        unscaled_df = self.get_unscaled_data(df)
        return unscaled_df[base_name]

    def get_unscaled_data(self, df=None):
        if df is None:
            df = self.df.loc[:, self.scaled_features]
        if self.minmax_scaler is None:
            raise Exception("Trying to rescale data without prior normalization")

        return self.__inverse_transform_columns__(df)
        return pd.DataFrame(data=self.minmax_scaler.inverse_transform(
                                    df.loc[:, features_list]),
                                columns=features_list)

    def __inverse_transform_columns__(self, scaled_df):        
        df_columns = self.scaled_features
        target_columns = scaled_df.columns.values

        temp_df = self.df.copy().loc[:, df_columns]
        temp_df[target_columns] = scaled_df

        temp_df = pd.DataFrame(self.minmax_scaler.inverse_transform(temp_df), columns=df_columns)
        temp_df.dropna(inplace=True)
        return temp_df.loc[:, target_columns]
        

    def compute_features(self, price_column='Close',predict_n=1, thresh_diff=None):
        base_feature_names = self.df.columns.values

        self.df['LowLen'] = self.df.apply(lambda r: np.minimum(r['Open'], r[price_column]) - r['Low'], axis=1)
        self.df['RSI(14)'] = ta.momentum.RSIIndicator(self.df[price_column], window=14).rsi()
        self.df['GAP'] = compute_GAP(self.df[price_column], self.df['Open'])
        self.df['RSI_diff'] = self.df['RSI(14)'].diff(periods=predict_n)
        self.df['Volume_diff'] = self.df['Volume'].diff(periods=predict_n)


        macd = ta.trend.MACD(self.df[price_column])
        self.df['MACD'] = macd.macd()
        self.df['MACD_diff'] = macd.macd_diff()
        self.df['MACD_signal'] = macd.macd_signal()

        self.df['BodyLen'] = (self.df[price_column] - self.df['Open']).abs()

        bg_band = ta.volatility.BollingerBands(self.df[price_column])
        self.df['BG_L_Band'] = bg_band.bollinger_lband()
        self.df['BG_H_Band'] = bg_band.bollinger_hband()
        self.df['BG_L_Band_Indicator'] = bg_band.bollinger_lband_indicator()
        self.df['BG_H_Band_Indicator'] = bg_band.bollinger_hband_indicator()
        

        self.df['ROC'] = ta.momentum.ROCIndicator(self.df[price_column]).roc()
        self.df['StochOsc'] = ta.momentum.StochasticOscillator(self.df['High'], self.df['Low'], self.df[price_column]).stoch()


        sma_10 = ta.trend.SMAIndicator(self.df[price_column], window=10).sma_indicator()
        sma_20 = ta.trend.SMAIndicator(self.df[price_column], window=20).sma_indicator()
        self.df['SMA(10)'] = sma_10
        self.df['SMA(20)'] = sma_20
        
        ema = ta.trend.EMAIndicator(self.df[price_column], window=14).ema_indicator()
        self.df['EMA(14)'] = ema
        self.df['EMA_Diff'] = ema.diff(periods=predict_n)
        self.df['SMA(20) - SMA(10)'] = sma_20 - sma_10

        self.df['Difference'] = self.df[price_column].diff(periods=predict_n)
        self.df['PercentageDiff'] = self.df[price_column].pct_change(periods=predict_n) 

        if thresh_diff is None:
            self.df['Tendency'] = compute_tendency(self.df[price_column], percentage=True, labels=['lower', 'higher'])
        else:
            self.df['Tendency'] = compute_tendency(self.df[price_column], percentage=True, thresh_diff=thresh_diff, labels=['lower','stay', 'higher'])
        
        self.df['NextPrice'] = self.df[price_column].shift(periods=-predict_n)
        self.df['Next'] = self.df['Tendency'].shift(periods=-predict_n)


        feature_names = [col for col in self.df.columns.values if col not in base_feature_names]        
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.feature_names = feature_names
        return feature_names

    @classmethod
    def merge_datasets(cls, ds_a, ds_b):
        ds_a.append_dataset(ds_a, ds_b)
        return ds_a

    @classmethod
    def read_from(cls, file_path, delimiter=' ', trim_column_names=True, compute_features=False, predict_n=1, thresh_diff=None, normalize=False):
        df = pd.read_csv(file_path, delimiter=delimiter)
        # set the index to start at 0 in case it is at 1 in the data
        df.index = np.arange(0, len(df))
        if trim_column_names:
            trim_columns(df)

        dataset = cls(df)
        if compute_features: 
            dataset.compute_features(predict_n=predict_n, thresh_diff=thresh_diff) 
        if normalize:
            dataset.normalize_data()

        return dataset

    @classmethod
    def import_folder(cls, folder_path, files_pattern='*.txt', recursive=False,
             compute_features=False, predict_n=1, normalize=False):

        files = Path(folder_path)
        files = files.rglob(files_pattern) if recursive else files.glob(files_pattern)

        datasets_dict = {
            cls.read_from(file, compute_features=compute_features, predict_n=predict_n, normalize=normalize):file.name 
            for file in files
        }


        return datasets_dict



if __name__ == '__main__':
    dataset = StocksDataWrapper.read_from('./data/AAPL.txt')

    dataset['Difference'] = dataset['Close'].diff()
    dataset['Diff4'] = dataset['Close'].diff(periods=4)
    dataset['DiffPercent'] = dataset['Close'].pct_change(periods=4) 
    dataset['Tendency'] = compute_tendency(dataset['Close'].pct_change(), labels=['lower', 'higher'])
    dataset['RSI(14)'] =  ta.momentum.RSIIndicator(dataset['Close'], window=14).rsi()
    dataset['GAP'] = compute_GAP(dataset['Close'], dataset['Open'])
    #print(dataset.head())
    dataset.normalize_data(features_list=['Open', 'Close', 'Volume'])
    #print(dataset.get_unscaled_data())


    dataset = StocksDataWrapper.read_from('./data/AAPL.txt')
    dataset.compute_features(predict_n=5)
    splits = dataset.get_datasets(sequences=True)
    print(splits)
    
    #print(f"Final dataset :\r\n ", dataset.head(20))

