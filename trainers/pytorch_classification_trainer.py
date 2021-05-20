from trainers.pytorch_base_trainer import PytorchTrainer
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score

class PytorchClassificationTrainer(PytorchTrainer):
    def __init__(self, model, device, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        PytorchTrainer.__init__(self, model, device, use_wandb=use_wandb, project_label=project_label, verbose=verbose)

    def compute_loss(self, loss_fn, output, target):
        return loss_fn(output, target.long())

    def compute_metrics(self, output, target):
        metrics_dict = {}
        # take the class with the highest score
        output = output.argmax(1)
        metrics_dict['accuracy'] = np.sum(output==target) / len(target)

        # if binary classification problem
        if len(np.unique(target)) == 2:
            fpr, tpr, thresholds = roc_curve(target, output)
            metrics_dict['roc_auc'] = auc(fpr, tpr)

        metrics_dict['f1_score'] = f1_score(target, output, average='weighted')

        return metrics_dict

  
