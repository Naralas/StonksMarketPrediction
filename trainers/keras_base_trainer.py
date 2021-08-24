from trainers.base_trainer import BaseTrainer
import wandb
from wandb.keras import WandbCallback
import numpy as np


class KerasTrainer(BaseTrainer):
    """Base trainer class for Keras models.

    Args:
        BaseTrainer (BaseTrainer object): Inheriting from the top abstract class which mainly sets up wandb. 
    """
    def __init__(self, model, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        """
        Init function of the trainer, will call the BaseTrainer.__init__(...)
        Args:
            model (Keras model object, inheriting models.base_model.BaseModel in the project): 
            use_wandb (bool, optional): Flag to use weights and biases for tracking metrics. You will have to create an API key. Defaults to True.
            project_label (str, optional): Project label, used by wandb. Defaults to "Default_Project_Name".
            verbose (bool, optional): Verbosity, will print metrics if true. Defaults to False.
        """
        BaseTrainer.__init__(self, model, use_wandb=use_wandb, project_label=project_label, verbose=verbose)

    def train(self, train_set, val_set=None):
        """Training function for the keras model

        Args:
            train_set (Keras dataset or numpy array): Training set.
            val_set (Keras dataset or numpy array, optional): [description]. Validation set. Defaults to None.

        Returns:
            Python dict: If a validation set was provided, returns a python dict of metrics with keys being the label of the metric and values, the metric value.
        """
        model = self.model

        callbacks = []
        if self.use_wandb:
            callbacks.append(WandbCallback())

        history = model.fit(train_set, validation_data=val_set, epochs=model.config['n_epochs'], 
            shuffle=False, verbose=self.verbose, callbacks=callbacks)
        
        metrics = {}
        
        if val_set is not None:
            output, target = self.predict(val_set)
            metrics = self.compute_metrics(output, target)  
            if self.use_wandb:
                wandb.log(metrics)
                wandb.finish()
            if self.verbose:
                print(f"Metrics : {metrics}")

        return metrics

    def predict(self, dataset):
        """Make predictions on the data passed.

        Args:
            dataset (Keras dataset): Samples to make predictions on.

        Returns:
            Tuple(numpy array, numpy array): Tuple of numpy arrays containing the predictions and labels of the dataset
        """
        predictions = np.array([])
        labels = np.array([])

        for x, y in dataset:
            predictions = np.concatenate([predictions, self.model.predict(x).flatten()])
            labels = np.concatenate([labels,y])
    
        return predictions, labels
    
    def evaluate(self, dataloader):
        """Computes the metrics for the data given.

        Args:
            dataloader (Keras dataset): Samples.

        Returns:
            Python dict: Python dict of metrics with keys being the label of the metric and values, the metric value.
        """
        predictions, labels = self.predict(dataloader)
        return self.compute_metrics(predictions, labels)


    def compute_metrics(self, output, target):
        raise NotImplementedError()