from trainers.keras_base_trainer import KerasTrainer
from helpers.plots_helper import plot_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb 
import numpy as np
import matplotlib.pyplot as plt


class KerasRegressionTrainer(KerasTrainer):
    def __init__(self, model, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        KerasTrainer.__init__(self, model, use_wandb=use_wandb, project_label=project_label, verbose=verbose)
    
    def compute_metrics(self, output, target):
        metrics_dict = {}

        ax = plot_predictions(target, output)
        metrics_dict['prices_plot'] = wandb.Image(plt)
        plt.close()

        return metrics_dict