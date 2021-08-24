from trainers.pytorch_base_trainer import PytorchTrainer
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, f1_score

class PytorchClassificationTrainer(PytorchTrainer):
    """Pytorch classification trainer class. Defines specific behaviors for classification (such as softmax output layer) and metrics.

    Args:
        PytorchTrainer (PytorchTrainer): : Inherited top class.
    """
    def __init__(self, model, device, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        """
        Init class of the trainer. Will call PytorchTrainer.__init__(...).
        
        Args:
            model (Keras model object, inheriting models.base_model.BaseModel in the project): 
            device (string) : 'cuda' or 'cpu', train on GPU or CPU.
            use_wandb (bool, optional): Flag to use weights and biases for tracking metrics. You will have to create an API key. Defaults to True.
            project_label (str, optional): Project label, used by wandb. Defaults to "Default_Project_Name".
            verbose (bool, optional): Verbosity, will print metrics if true. Defaults to False.
        """
        PytorchTrainer.__init__(self, model, device, use_wandb=use_wandb, project_label=project_label, verbose=verbose)

    def compute_loss(self, loss_fn, output, target):
        """Classification implementation function to compute loss. Will set target to tensor of longs.

        Args:
            loss_fn (Pytorch loss function): Pytorch loss function object.
            output (Pytorch Tensor): Pytorch tensor containing the model output (predictions).
            target (Pytorch Tensor): Pytorch tensor containing the model targets (labels).

        Returns:
            Pytorch Tensor : Tensor of loss.
        """

        return loss_fn(output, target.long())

    def compute_metrics(self, output, target):
        """Compute classification metrics for the data given. At the moment, accuracy and f1, and roc_auc if binary class.

        Args:
            output (Numpy array): Numpy array of model output (predictions).
            target (Numpy array): Numpy array of labels (true values).

        Returns:
            Python dict: Python dict of metrics with keys being the metric label and values the metric value. 
        """
        metrics_dict = {}
        # take the class with the highest score
        output = output.argmax(1)
        metrics_dict['acc'] = np.sum(output==target) / len(target)

        # if binary classification problem
        if self.model.config['output_dim'] == 2:
            fpr, tpr, thresholds = roc_curve(target, output)
            metrics_dict['roc_auc'] = auc(fpr, tpr)


        metrics_dict['f1'] = f1_score(target, output, average='weighted')

        return metrics_dict

    def predict(self, dataloader):
        """Make predictions on data given. Will create a model with added softmax output layer to compute class probabilities.

        Args:
            dataloader (Pytorch dataloader): Pytorch dataloader object.

        Returns:
            Tuple(numpy array, numpy array): Tuple of numpy arrays containing the predictions and labels of the dataset
        """
        # add probability output with softmax to output array with sum to 1 for the classes
        probability_model = nn.Sequential(
            *self.model.model,
            nn.Softmax(1),
        )
        self.model.eval()
        with torch.no_grad():
            predictions = []
            labels = []

            for data, target in dataloader:
                data = data.to(self.device)
                output = probability_model(data)  
                predictions.extend(output.cpu().numpy())
                labels.extend(target.numpy())

        return np.squeeze(np.array(predictions)), np.array(labels)
