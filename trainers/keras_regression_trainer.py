from trainers.keras_base_trainer import KerasTrainer
from helpers.plots_helper import plot_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import wandb 
import numpy as np
import matplotlib.pyplot as plt


class KerasRegressionTrainer(KerasTrainer):
    """Keras regression trainer subclass. Defines specific behaviors for regression such as  metrics.

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
        """Compute regression metrics for the data given. At the moment, Mean Average Error (MSE), Mean Average Percentage Error (MAPE) and Mean Squared Error (MSE).

        Args:
            output (Numpy array): Numpy array of model output (predictions).
            target (Numpy array): Numpy array of labels (true values).

        Returns:
            Python dict: Python dict of metrics with keys being the metric label and values the metric value. 
        """
        metrics_dict = {}

        metrics_dict['mse'] = mean_squared_error(target, output)
        metrics_dict['mae'] = mean_absolute_error(target, output)
        metrics_dict['mape'] = mean_absolute_percentage_error(target, output)

        return metrics_dict

    def predict(self, dataloader):
        """Make predictions on data given. Creates a model with added SoftMax output layer to compute class probabilities.

        Args:
            dataset (Keras dataset): Keras dataset of samples.

        Returns:
            Tuple(numpy array, numpy array): Tuple of numpy arrays containing the predictions and labels of the dataset
        """
        predictions = np.array([])
        labels = np.array([])
        for x, y in dataloader:
            predictions = np.concatenate([predictions, self.model.predict(x).flatten()])
            labels = np.concatenate([labels,y])
        return predictions, labels