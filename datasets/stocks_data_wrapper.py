import pandas as pd
import numpy as np
import ta

from pathlib import Path
from sklearn import preprocessing
from helpers.data_helper import *


default_features_list = ['Close', 'Volume', 'MACD_diff', 'RSI(14)', 'PercentageDiff', 'LowLen', 'RSI_diff']


class StocksDataWrapper:
    """Pandas DataFrame wrapper for stocks data
    all columns of the dataframe can be accessed with the array notation for example : data_wrapper['Close']

    This class contains functions for reading from data files in CSV format, compute features, 
    normalize/de-normalize and build machine-learning ready datasets.

    The usual pipeline for getting data is the following set of functions :

    For a single stock :
    *StocksDataWrapper.read_from(path:str, [delimiter:char], [predict_n:int], [trim_column_names:boolean]
                                [compute_features:boolean], [thresh_diff:float], [normalize:boolean]) - class function, read data and build object 
                                                                                                        returns a StocksDataWrapper object
    
    Or for reading all in a folder :
    *StocksDataWrapper.import_folder(cls, path:str, [files_pattern:str], [recursive:boolean],
                                [compute_features:boolean], [thresh_diff:float], [normalize:boolean]) - class function, 
                                                                                                        returns a dict [file_name:StocksDataWrapper]
    If not done previously at the reading step :
    *compute_features([price_column:str], [predict_n=int], [thresh_diff:float])                       - add technical indicators features and trends
                                                                                                        returns the list of features as a list [str]
    For normalizing, if not done previously :
    *normalize_data([features_list:list], [scale:tuple(float,float)], [inplace:boolean])              - normalization, if no features given by default will normalize all numerical columns

    *get_datasets([n_splits:int], [val_size:float], [sequences:boolean], [seq_len:int],               - build datasets, can be done in sequences (eg for LSTM)
                                [y_column:str], [negative_labels:boolean],                              negative labels : some DL frameworks do not accept -1 as label, start at 0 for lower
                                [features_list:list[str]])                                              returns a numpy array of tuple of datasets splits (in timeseries manner) :
                                                                                                        (X_train, X_val, X_test, y_train, y_val, y_test) or without val objects if val_size is 0.0

    There are other functions available such as de-normalization, merging datasets, etc.
    """
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
            self.df[col] = self.df[col].pct_change()


    def normalize_data(self, features_list=[], scale=(0,1), inplace=True):
        """Normalize the columns passed to the given scale.

        Args:
            features_list (list, optional): Features list to normalize. Defaults to [] -> will use default features list
            scale (tuple, optional): Normalization values scale. Defaults to (0,1).
            inplace (bool, optional): Replace the values in the object's dataframe or not. Defaults to True.

        Returns:
            Pandas.DataFrame: normalized dataframe if inplace is set to False 
        """
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
                        negative_labels=False,
                        features_list=default_features_list):
        """Build machine learning-ready datasets, with timeseries cross-validation if n_splits > 1

        Args:
            n_splits (int, optional): Number of time series split. Defaults to 5.
            val_size (float, optional): Size of validation sets, can be set to 0 for no sets. Defaults to 0.3.
            sequences (bool, optional): Build data as sequences : [[Feature_A_t-1, ...][Feature_A_t-2, ...], ...]. Can be used for LSTM. Defaults to False.
            seq_len (int, optional): Size of sequences. Defaults to 5.
            y_column (str, optional): Target column for ML datasets. Defaults to 'NextPrice'.
            negative_labels (bool, optional): Set to start labels at -1 or 0, as some DL frameworks don't accept negative values. Defaults to False.
            features_list ([type], optional): List of features to include in the training data. Defaults to default_features_list.

        Returns:
            numpy.array: Numpy array of the splits of datasets. Each element of the array is a tuple in the following format :
            (X_train, X_val, X_test, y_train, y_val, y_test) if val_size > 0.0
            (X_train, X_test, y_train, y_test) otherwise
        """

        dataset = self.df.copy()

        # filter only the features columns in the data and the target column
        features_list = features_list + [y_column] if y_column not in features_list else features_list

        dataset = dataset.loc[:, features_list]


        # replace the textual data (tendencies) by numerical values
        for col in dataset.columns:
            if len(np.unique(dataset[y_column])) > 2:
                if negative_labels:
                    dataset[col] = dataset[col].replace({'higher':1, 'stay':0, 'lower':-1})
                else:
                    dataset[col] = dataset[col].replace({'higher':2, 'stay':1, 'lower':0})
            else:
                dataset[col] = dataset[col].replace({'higher':1, 'lower':0})

        # remove the target column and build sequences if needed
        X = dataset.loc[:, dataset.columns != y_column].values
        if sequences:
            # X.shape = (X_a, X_b) -> (X_a, seq_len, X_b)
            X = build_sequences(X, seq_len=seq_len)

        y = dataset[y_column].values[:X.shape[0]]
        
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
        
    def __inverse_transform_columns__(self, scaled_df):        
        df_columns = self.scaled_features
        target_columns = scaled_df.columns.values

        temp_df = self.df.copy().loc[:, df_columns]
        temp_df[target_columns] = scaled_df

        temp_df = pd.DataFrame(self.minmax_scaler.inverse_transform(temp_df), columns=df_columns)
        temp_df.dropna(inplace=True)
        return temp_df.loc[:, target_columns]
        

    def compute_features(self, price_column='Close',predict_n=1, thresh_diff=None):
        """Compute a list of technical indicators and trends of the prices and volumes history of the stock

        Args:
            price_column (str, optional): Column name that will be used as the main price. Defaults to 'Close'.
            predict_n (int, optional): Number of periods difference that will be predicted and used for target columns. Defaults to 1.
            thresh_diff ([type], optional): Threshold of price difference that will be used for the stationary class . Defaults to None -> no stationary class.

        Returns:
            list[str]: list of feature columns that were included 
        """
        base_feature_names = self.df.columns.values

        self.df['LowLen'] = self.df.apply(lambda r: np.minimum(r['Open'], r[price_column]) - r['Low'], axis=1)
        self.df['RSI(14)'] = ta.momentum.RSIIndicator(self.df[price_column], window=14).rsi()
        self.df['GAP'] = compute_GAP(self.df[price_column], self.df['Open'])
        self.df['RSI_diff'] = self.df['RSI(14)'].diff(periods=1)
        self.df['Volume_diff'] = self.df['Volume'].diff(periods=1)


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
        self.df['EMA_Diff'] = ema.diff(periods=1)
        self.df['SMA(20) - SMA(10)'] = sma_20 - sma_10

        self.df['Difference'] = self.df[price_column].diff(periods=1)
        self.df['PercentageDiff'] = self.df[price_column].pct_change(periods=1) 

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
        """Read stock data from a given file. Can compute features and normalize already.

        Args:
            file_path ([type]): Path to the file that will contain data in CSV format
            delimiter (str, optional): Values delimiter of the CSV file. Defaults to ' '.
            trim_column_names (bool, optional): Will remove prefix in the column names, e.g. AAPL.Close -> Close. Defaults to True.
            compute_features (bool, optional): Return data wrapper with already computed features. Defaults to False.
            predict_n (int, optional): N days prediction, used for features computation. Defaults to 1.
            thresh_diff ([type], optional): Price difference threshold for stationary class, used for features computation. Defaults to None.
            normalize (bool, optional): Normalize the values. Defaults to False.

        Returns:
            StocksDataWrapper: Stocks data wrapper object with inner pandas dataframe containing the data
        """
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
        """Helper method to read all data from a directory

        Args:
            folder_path ([type]): Path to folder containing data
            files_pattern (str, optional): Matching pattern of the files that will be read. Defaults to '*.txt'.
            recursive (bool, optional): Recursively search for stocks files. Defaults to False.
            compute_features (bool, optional): Return data wrappers with features already computed. Defaults to False.
            predict_n (int, optional): N days predictions, used for features computation. Defaults to 1.
            normalize (bool, optional): Normalization. Defaults to False.

        Returns:
            dictionary(file_name:StocksDataWrapper): Dictionary of read files with keys being the file's name and values the matching StocksDataWrapper object 
        """

        files = Path(folder_path)
        files = files.rglob(files_pattern) if recursive else files.glob(files_pattern)

        datasets_dict = {
            file.name:cls.read_from(file, compute_features=compute_features, predict_n=predict_n, normalize=normalize)
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
    dataset.normalize_data(features_list=['Open', 'Close', 'Volume'])


    dataset = StocksDataWrapper.read_from('./data/AAPL.txt')
    dataset.compute_features(predict_n=5)
    splits = dataset.get_datasets(sequences=True)
    print(splits)

