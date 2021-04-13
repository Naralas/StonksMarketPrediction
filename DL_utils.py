import torch
import torch.nn as nn
import wandb 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc, f1_score

def create_model(model_class, device, input_dim, output_dim, 
            loss_fn, optimizer, lr, 
            use_wandb=True, project_label=None, **kwargs):

    model = model_class(input_dim, output_dim, **kwargs)
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)
    loss_fn = loss_fn()
    config_dict = dict(
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        lr=lr,
    )
    if use_wandb:
        init_wandb(model, config_dict, project_label)

    return model, optimizer, loss_fn


def init_wandb(model, config_dict, project_label):
    wandb.init(project=f"{project_label}", config=config_dict)
    wandb.watch(model)

def train(model, dataloader, n_epochs, optimizer, loss_fn, device, use_wandb=True):
    if use_wandb:
        wandb.config.update({'n_epochs':n_epochs})
    
    model.train()

    for epoch in range(n_epochs):
        epoch_acc = 0
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(data)
            
            loss = loss_fn(output, target.unsqueeze(1))
            acc = binary_acc(output, target.unsqueeze(1))
            epoch_acc += acc.item()
            
            loss.backward()
            optimizer.step()

        acc = epoch_acc / len(dataloader)
        if use_wandb:
            wandb.log({"train_loss": loss.item(), "train_accuracy":acc})
        print(f"Epoch {epoch+1}, loss: {loss.item()}, accuracy : {acc:.2f}")


def predict(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        predictions = []
        labels = []
        for data, target in dataloader:
            data = data.to(device)
            output = model(data) # calling model calls forward function
            prediction = torch.round(torch.sigmoid(output))
            predictions.extend(prediction.cpu().numpy())
            labels.extend(target.numpy())
    return np.array(predictions), np.array(labels)


def binary_acc(output, target):
    # https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
    output_tag = torch.round(torch.sigmoid(output))

    correct_results_sum = (output_tag == target).sum().float()
    acc = correct_results_sum/target.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def compute_metrics(model, dataloader, device, labels=None, predictions=None):
    metrics_dict = {}
    if labels is None or predictions is None:
        predictions, labels = predict(model, dataloader, device)

    metrics_dict['accuracy'] = accuracy_score(predictions, labels)
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    metrics_dict['roc_auc'] = auc(fpr, tpr)
    metrics_dict['f1_score'] = f1_score(labels, predictions)


    return metrics_dict
