# StonksMarketPrediction

Github repo for the master thesis project of Ludovic Herbelin, supervised by Jacques Savoy.

Swiss Joint Master of Science in Computer Science, March 2021 - September 2021

![](https://mcs.unibnf.ch/wp-content/uploads/2018/03/logo-transp-e1531733843534.png)


The structure of the project and some of the files such as the base classes were inspired by this [github project](https://github.com/Ahmkel/Keras-Project-Template).

[Stonks ?](https://knowyourmeme.com/memes/stonks)

## Goals

The goal of this project is to work on financial markets, try and predict the changes and / or tendencies of the stock prices daily. We will compare the performances of the different models with the efficient market hypothesis by building a model predicting the trends and prices according to a random walk process.

## Project structure

```
├───.gitignore                                  `
├───README.md
├───requirements.txt
│   
├───data                 - data in csv-like format files
│   ├───AAL.txt
│   ├───AAPL.txt
│   ├───AMZN.txt
│   ├───CMCSA.txt
│   ├───COST.txt
│   ├───GM.txt
│   ├───GOOG.txt
│   ├───IBM.txt
│   ├───JNJ.txt
│   ├───KO.txt
│   ├───PEP.txt
│   ├───TSLA.txt
│   ├───WMT.txt
│   └───XOM.txt
│       
├───datasets                       - datasets files, for pytorch and wrapper with pandas dataframe
│   ├───stocks_data_wrapper.py     - wrapper with a pandas dataframe for stocks data, compute features, normalization, etc.
│   └───torch_datasets.py
│       
├───helpers                        - general helper files : plots, preprocessing utils, etc.
│   ├───data_helper.py             - data-related functions compute some features, split data, etc.
│   ├───learning_utils.py
│   ├───plots_helper.py            - help plot the data and results with pyplot
│   ├───preprocessor.py
│   └───__init__.py
│       
├───models                         - model classes for pytorch and keras
│   ├───base_model.py              - abstract basic model by the different sub-models
│   ├───keras_lstm_model.py
│   ├───pytorch_linear_model.py
│   └───__init__.py
│       
├───notebooks                       - jupyter notebooks with the different experiments
│   ├───FeaturesComparison.ipynb
│   ├───KerasLSTMClassification.ipynb
│   ├───KerasLSTMRegressionSplits.ipynb
│   ├───notebook_config.py          - helper for notebooks settings (path)
│   ├───pipelines                   - pipelines to run on all stocks
│       ├───notebook_config.py
│       ├───Pipeline_KerasLSTM.ipynb
│       ├───Pipeline_ML_Classification.ipynb
│       ├───Pipeline_ML_Regression.ipynb
│       ├───Pipeline_Pytorch_LinearModel_Prediction.ipynb
│       ├───Pipeline_RandomWalk.ipynb
│       ├───results                 - results output folder
│           └───...
│   ├───PCA.ipynb
│   ├───Pytorch_LinearModel_Prediction.ipynb
│   ├───Pytorch_LinearModel_Regression.ipynb
│   ├───Pytorch_LSTMModel_StockPrediction.ipynb
│   ├───RandomWalk.ipynb
│   ├───StockPriceRegressionModel.ipynb
│   ├───StockPriceTendencyPrediction.ipynb
│   └───__init__.py
│       
└───trainers                - trainer files which handle the models training and predictions for pytorch and keras
    ├───base_trainer.py     - base trainer class which is inherited by all sub-trainers for the different frameworks
    ├───keras_base_trainer.py      
    ├───keras_classification_trainer.py
    ├───keras_regression_trainer.py
    ├───pytorch_base_trainer.py
    ├───pytorch_classification_trainer.py
    ├───pytorch_regression_trainer.py
    └───__init__.py

```

## Setup

Clone the repo wherever you want, then create a virtual environment and install the modules :

- `python3 -m venv <path/to/virtualenv>`
- Depending on your OS : `source <path/to/virtualenv>/bin/activate` for Unix or `<path\to\virtualenv\activate.bat>` for Windows
- `pip install -r requirements.txt`
- `jupyter lab`

You will now be able to run the experiments in the `/notebooks/` folder.
