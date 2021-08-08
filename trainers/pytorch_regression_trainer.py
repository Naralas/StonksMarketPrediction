from trainers.pytorch_base_trainer import PytorchTrainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class PytorchRegressionTrainer(PytorchTrainer):
    def __init__(self, model, device, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        PytorchTrainer.__init__(self, model, device, use_wandb=use_wandb, project_label=project_label, verbose=verbose)

    def compute_loss(self, loss_fn, output, target):
        return loss_fn(output.squeeze(), target)

    def compute_metrics(self, output, target):
        metrics_dict = {}
        output = output.squeeze()
        metrics_dict['mse'] = mean_squared_error(target, output)
        metrics_dict['mae'] = mean_absolute_error(target, output)
        metrics_dict['mape'] = mean_absolute_percentage_error(target, output)
        return metrics_dict
