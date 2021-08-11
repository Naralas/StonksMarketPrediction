from trainers.pytorch_base_trainer import PytorchTrainer
import torch
import numpy as np
import torch.nn as nn
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
        metrics_dict['acc'] = np.sum(output==target) / len(target)

        # if binary classification problem
        if self.model.config['output_dim'] == 2:
            fpr, tpr, thresholds = roc_curve(target, output)
            metrics_dict['roc_auc'] = auc(fpr, tpr)


        metrics_dict['f1'] = f1_score(target, output, average='weighted')

        return metrics_dict

    def predict(self, dataloader):
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
