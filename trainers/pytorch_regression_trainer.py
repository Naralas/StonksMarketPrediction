from trainers.pytorch_base_trainer import PytorchTrainer
from sklearn.metrics import mean_squared_error, r2_score

class PytorchRegressionTrainer(PytorchTrainer):
    def __init__(self, model, device, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        PytorchTrainer.__init__(self, model, device, use_wandb=use_wandb, project_label=project_label, verbose=verbose)


    def compute_metrics(self, output, target):
        metrics_dict = {}
        metrics_dict['mse'] = mean_squared_error(target, output)
        metrics_dict['r2_score'] = r2_score(target, output)
        return metrics_dict
