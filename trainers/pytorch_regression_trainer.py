from trainers.pytorch_base_trainer import PytorchTrainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class PytorchRegressionTrainer(PytorchTrainer):
    """Pytorch regression trainer subclass.

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
        """Regression implementation function to compute loss.

        Args:
            loss_fn (Pytorch loss function): Pytorch loss function object.
            output (Pytorch Tensor): Pytorch tensor containing the model output (predictions).
            target (Pytorch Tensor): Pytorch tensor containing the model targets (labels).

        Returns:
            Pytorch Tensor : Tensor of loss.
        """
        return loss_fn(output.squeeze(), target)

    def compute_metrics(self, output, target):
        """Compute regression metrics for the data given. At the moment, Mean Average Error (MSE), Mean Average Percentage Error (MAPE) and Mean Squared Error (MSE).

        Args:
            output (Numpy array): Numpy array of model output (predictions).
            target (Numpy array): Numpy array of labels (true values).

        Returns:
            Python dict: Python dict of metrics with keys being the metric label and values the metric value. 
        """
        metrics_dict = {}
        output = output.squeeze()
        metrics_dict['mse'] = mean_squared_error(target, output)
        metrics_dict['mae'] = mean_absolute_error(target, output)
        metrics_dict['mape'] = mean_absolute_percentage_error(target, output)
        return metrics_dict
