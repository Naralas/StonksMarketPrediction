from trainers.base_trainer import BaseTrainer
import wandb
import torch


class PytorchTrainer(BaseTrainer):
    def __init__(self, model, device, use_wandb=True, project_label="Default_Project_Name", verbose=False):
        BaseTrainer.__init__(self, model, use_wandb=use_wandb, project_label=project_label, verbose=verbose)
        self.device = device

    def train(self, dataloader, debug=False):
        model = self.model
        device = self.device 

        optimizer = self.model.optimizer
        loss_fn = self.model.loss_fn

        
        model.train()
        if debug:
            torch.autograd.set_detect_anomaly(True)

        if self.use_wandb:
            wandb.watch(model)
        
        for epoch in range(model.config['n_epochs']):
            epoch_metrics = {}
            for data, target in dataloader:
                # init the data and gradients
                data = data.to(device)
                target = target.to(device).unsqueeze(1)
                optimizer.zero_grad()

                # make predictions
                output = model(data)

                # compute loss and other metrics, backprop
                loss = loss_fn(output, target)
                
                batch_metrics = self.compute_metrics(output, target)
                
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

    def predict(self, dataloader):
        model = self.model
        model.eval()
        with torch.no_grad():
            predictions = []
            labels = []

            # TODO : compute metrics with numpy / regular values and update in trainer function
            for data, target in dataloader:
                data = data.to(device)
                output = model(data) # 
                predictions.extend(prediction.cpu().numpy())
                labels.extend(target.numpy())

        return np.array(predictions), np.array(labels)
    
    def evaluate(self, dataloader):
        model = self.model
        model.eval()

        with torch.no_grad():
            metrics = {}
            for data, target in dataloader:
                data = data.to(self.device)
                output = model(data) # calling model calls forward function
                
                batch_metrics = self.compute_metrics(output, target)

                for k, v in batch_metrics.items():
                    # if key does not exist set to empty list and append, if exists, append
                    metrics.setdefault(k, []).append(v)


            for metric, values in metrics.items():
                metrics[metric] = sum(values) / len(values)

            return metrics


    def compute_metrics(self, output, target):
        raise NotImplementedError()