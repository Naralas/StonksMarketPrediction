from trainers.keras_base_trainer import KerasTrainer
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import wandb 
import numpy as np
import matplotlib.pyplot as plt

from helpers.plots_helper import plot_heatmap



class KerasClassificationTrainer(KerasTrainer):
    def __init__(self, model, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        KerasTrainer.__init__(self, model, use_wandb=use_wandb, project_label=project_label, verbose=verbose)
    
    def compute_metrics(self, output, target):
        metrics_dict = {}
        # take the argmax (index of max {0;1;2} or {0;1} for binary classification)
        output = np.argmax(output, axis=1)

        metrics_dict['accuracy'] = accuracy_score(target, output)

        class_labels = ['higher', 'stay', 'lower']

        # if binary classification problem
        if len(np.unique(target)) == 2:
            fpr, tpr, thresholds = roc_curve(target, output)
            metrics_dict['roc_auc'] = auc(fpr, tpr)
            class_labels = ['higher', 'lower']

        metrics_dict['f1_score'] = f1_score(target, output, average='weighted')

        ax = plot_heatmap(labels=target, predictions=output, class_labels=class_labels)
        metrics_dict['confusion_matrix'] = wandb.Image(plt)
        plt.close()

        return metrics_dict
    
    def predict(self, dataset):
        # probability model from https://www.tensorflow.org/tutorials/keras/classification
        # attach softmax to convert to probabilities
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

        predictions = []
        labels = np.array([])

        for x, y in dataset:
            predictions.extend(probability_model.predict(x))
            labels = np.concatenate([labels,y])
    
        return np.array(predictions), labels