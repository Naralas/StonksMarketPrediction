from trainers.base_trainer import BaseTrainer
import wandb
import torch
import numpy as np


class PytorchTrainer(BaseTrainer):
    """Pytorch base trainer class for regression and classification models.

    Args:
        BaseTrainer (BaseTrainer object): Inheriting from the top abstract class which mainly sets up wandb. 
    """
    def __init__(self, model, device, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        """
        Init function of the trainer, will call the BaseTrainer.__init__(...)
        Args:
            model (Keras model object, inheriting models.base_model.BaseModel in the project): 
            use_wandb (bool, optional): Flag to use weights and biases for tracking metrics. You will have to create an API key. Defaults to True.
            project_label (str, optional): Project label, used by wandb. Defaults to "Default_Project_Name".
            verbose (bool, optional): Verbosity, will print metrics if true. Defaults to False.
        """
        BaseTrainer.__init__(self, model, use_wandb=use_wandb, project_label=project_label, verbose=verbose)
        self.device = device

    def train(self, dataloader, debug=False):
        """Call this function to train the model on data given.

        Args:
            dataloader (Pytorch Dataloader object): Pytorch dataloader containing training samples. 
            debug (bool, optional): Debug flag, used to track null gradients, etc. Will slow down the training. Defaults to False.
        """
        model = self.model
        device = self.device 

        optimizer = self.model.optimizer
        loss_fn = self.model.loss_fn

        if debug:
            torch.autograd.set_detect_anomaly(True)

        if self.use_wandb:
            wandb.watch(model)
        
        for epoch in range(model.config['n_epochs']):
            epoch_metrics = {}
            for data, target in dataloader:
                model.train()
                # init the data and gradients
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()

                # make predictions
                output = model(data)

                # compute loss and other metrics, backprop
                loss = self.compute_loss(loss_fn, output, target)
                batch_metrics = self.compute_metrics(output.detach().cpu().numpy(), target.detach().cpu().numpy())
                
                # add these batch metrics to the epoch metrics
                for k, v in batch_metrics.items():
                    # if key does not exist set to empty list and append, if exists, append
                    epoch_metrics.setdefault(k, []).append(v)

                loss.backward()
                optimizer.step()

            for metric, values in epoch_metrics.items():
                epoch_metrics[metric] = sum(values) / len(values)

            if self.use_wandb:
                wandb.log(epoch_metrics)
            if self.verbose:
                print(f"Metrics : {epoch_metrics}")

        wandb.finish()

    def compute_loss(self, loss_fn, output, target):
        """Base trainer function to compute loss, can be overriden depending on loss (such as for adding softmax layer, etc.)

        Args:
            loss_fn (Pytorch loss function): Pytorch loss function object.
            output (Pytorch Tensor): Pytorch tensor containing the model output (predictions).
            target (Pytorch Tensor): Pytorch tensor containing the model targets (labels).

        Returns:
            Pytorch Tensor : Tensor of loss.
        """
        return loss_fn(output, target)

    def predict(self, dataloader):
        """Make predictions with the model on data given. Will set the model to eval and no gradients computation beforehand.

        Args:
            dataloader (Pytorch Dataloader): Dataloader of samples.

        Returns:
            Tuple (numpy array, numpy array): Tuple of numpy arrays containing predictions and labels.
        """
        model = self.model
        model.eval()
        with torch.no_grad():
            predictions = []
            labels = []

            for data, target in dataloader:
                data = data.to(self.device)
                output = model(data)  
                predictions.extend(output.cpu().numpy())
                labels.extend(target.numpy())

        return np.squeeze(np.array(predictions)), np.array(labels)
    
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