import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, roc_curve, auc

def plot_prices(date_series, price_series, quotation_name = "", ma_values = []):
    x_values = [datetime.datetime.strptime(d,"%Y-%m-%d").date() for d in date_series]
    ax = plt.gca()
    ax.set(title=f"{quotation_name} Stock Closing Price over time", ylabel='Closing price (USD)', xlabel='Time')

    formatter = mdates.DateFormatter("%m-%Y")

    ax.xaxis.set_major_formatter(formatter)

    ax.plot(x_values, price_series, label="Stock price")

    for val in ma_values:
        ax.plot(x_values, price_series.rolling(val).mean(), label=f"MA({val})")

    ax.legend(loc='best')

    return ax

def __label_density_hist__(ax, n, bins, x=4, y=0.01, r=0, **kwargs):
    """
    adapted from https://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
    """
    
    total = sum(n)

    # plot the label/text to each bin
    for i in range(0, len(n)):
        x_pos = (bins[i + 1] - bins[i]) / x + bins[i]
        y_pos = n[i] + (n[i] * y)
        
        if n[i] < 10:
            continue
        
        label = f"{(n[i] * 100.0 / total):.1f}%"
        ax.text(x_pos, y_pos, label, kwargs)

def plot_density_hist(values_series):
    ax = plt.gca()

    counts, bins, patches = ax.hist(values_series, 50, histtype='bar', ec='black')
    ax.tick_params(axis='x', rotation=90, labelsize=14)
    ax.set_xticks(bins)
    ax.set_title("Closing stock price differences")
    ax.set_xlabel('Price difference between current day and previous day (percentage of price)')
    ax.set_ylabel('Number of days')
    __label_density_hist__(ax, counts, bins, fontsize=14)

    return ax

def plot_normalized_histogram(series):
    ax = plt.gca()
    ax.set_title(f"Normalized histogram of series \"{series.name}\"")
    ax.hist(series, weights = np.ones_like(series) / len(series))
    return ax

def plot_cfm(clf, X_test, Y_test, normalize='true'):
    plot_confusion_matrix(clf, X_test, Y_test, normalize=normalize)

def plot_filtered_class_features(df, n_cols, feature_names, class_column):
    # get an array of subplots depending on the number of features, splitted into 3 columns
    fig, axs = plt.subplots(int(np.ceil(len(feature_names) / n_cols)), n_cols)
    fig.suptitle('Values of features (scaled) filtered by class')
    for i, feature in enumerate(feature_names):
        sub_df = df[[feature, class_column]]
        #print(f"[{i}] : {i // N_COLS}, {i % N_COLS}")
        for target_c in sub_df[class_column].value_counts().to_dict().keys():
            filtered_df = sub_df[sub_df[class_column] == target_c]
            # plot the histogram with the 
            axs[i // n_cols, i % n_cols].hist(filtered_df[feature], label=f"{feature} [{target_c}]")
            axs[i // n_cols, i % n_cols].legend(loc='best')


    return axs



def plot_heatmap(labels, predictions, class_labels, normalize='all'):
    cf_matrix = confusion_matrix(y_true=labels, y_pred=predictions, normalize=normalize)
    predict_df = pd.DataFrame(cf_matrix, index = [c for c in class_labels], columns = [c for c in class_labels])
    ax = sns.heatmap(predict_df, annot=True)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('True labels')

    return ax

def plot_roc_auc_curve(labels, predictions):
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    ax = plt.gca()
    ax.plot(fpr, tpr, color='orange', lw=4, label=f"ROC curve ({roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color='blue', lw=4, linestyle='dashed')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC AUC Curve')
    ax.legend(loc="best")
    return ax
