from trainers.keras_base_trainer import KerasTrainer
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt


class KerasClassificationTrainer(KerasTrainer):
    """Keras classification trainer subclass. Defines specific behaviors for classification (such as softmax output layer) and metrics.

    Args:
        KerasTrainer (KerasTrainer): Inherited top class.
    """
    def __init__(self, model, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        """
            Init class of the trainer. Will call KerasTrainer.__init__(...).
        Args:
            model (Keras model object, inheriting models.base_model.BaseModel in the project): 
            use_wandb (bool, optional): Flag to use weights and biases for tracking metrics. You will have to create an API key. Defaults to True.
            project_label (str, optional): Project label, used by wandb. Defaults to "Default_Project_Name".
            verbose (bool, optional): Verbosity, will print metrics if true. Defaults to False.
        """
        KerasTrainer.__init__(self, model, use_wandb=use_wandb, project_label=project_label, verbose=verbose)
    
    def compute_metrics(self, output, target):
        """Compute classification metrics for the data given. At the moment, accuracy and f1, but can add more in the future.

        Args:
            output (Numpy array): Numpy array of model output (predictions).
            target (Numpy array): Numpy array of labels (true values).

        Returns:
            Python dict: Python dict of metrics with keys being the metric label and values the metric value. 
        """
        metrics_dict = {}
        # take the argmax (index of max {0;1;2} or {0;1} for binary classification)
        output = np.argmax(output, axis=1)

        metrics_dict['acc'] = accuracy_score(target, output)
        metrics_dict['f1'] = f1_score(target, output, average='weighted')

        return metrics_dict
    
    def predict(self, dataset):
        """Make predictions on data given. Creates a model with added SoftMax output layer to compute class probabilities.

        Args:
            dataset (Keras dataset): Keras dataset of samples.

        Returns:
            Tuple(numpy array, numpy array): Tuple of numpy arrays containing the predictions and labels of the dataset
        """
        # probability model from https://www.tensorflow.org/tutorials/keras/classification
        # attach softmax to convert to probabilities
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

        predictions = []
        labels = np.array([])

        for x, y in dataset:
            predictions.extend(probability_model.predict(x))
            labels = np.concatenate([labels,y])
    
        return np.array(predictions), labels