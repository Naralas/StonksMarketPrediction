from trainers.base_trainer import BaseTrainer
import wandb
from wandb.keras import WandbCallback
import numpy as np


class KerasTrainer(BaseTrainer):
    def __init__(self, model, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        BaseTrainer.__init__(self, model, use_wandb=use_wandb, project_label=project_label, verbose=verbose)

    def train(self, train_set, val_set=None):
        model = self.model

        if self.use_wandb:
            history = model.fit(train_set, validation_data=val_set, epochs=model.config['n_epochs'], 
                    shuffle=False, verbose=self.verbose, callbacks=[WandbCallback()])

        else:
            history = model.fit(train_set, validation_data=val_set, epochs=model.config['n_epochs'], 
                    shuffle=False, verbose=self.verbose)
        
        output, target = self.predict(val_set)
        metrics = self.compute_metrics(output, target)

        if self.use_wandb:
            wandb.log(metrics)
            wandb.finish()
        if self.verbose:
            print(f"Metrics : {metrics}")

        return metrics

    def predict(self, dataset):
        predictions = np.array([])
        labels = np.array([])

        for x, y in dataset:
            predictions = np.concatenate([predictions, self.model.predict(x).flatten()])
            labels = np.concatenate([labels,y])
    
        return predictions, labels
    
    def evaluate(self, dataloader):
        predictions, labels = self.predict(dataloader)
        return self.compute_metrics(predictions, labels)


    def compute_metrics(self, output, target):
        raise NotImplementedError()