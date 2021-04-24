import torch
import torch.nn as nn
import wandb 
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc, f1_score, mean_squared_error

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

def binary_acc(output, target):
    # https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
    output_tag = torch.round(torch.sigmoid(output))

    correct_results_sum = (output_tag == target).sum().float()
    acc = correct_results_sum/target.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def cross_entropy_acc(output, target):
    return np.array([0])


def torch_mse(output, target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return mean_squared_error(target, output)

def train(model, dataloader, n_epochs, optimizer, loss_fn, device, 
            acc_func=binary_acc, acc_label='accuracy', epoch_log = 10, use_wandb=True):

    if use_wandb:
        wandb.config.update({'n_epochs':n_epochs})
    
    model.train()

    #torch.autograd.set_detect_anomaly(True)
    epochs_bar = tqdm(range(n_epochs))
    for epoch in epochs_bar:
        epoch_acc = 0
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device).unsqueeze(1)
            optimizer.zero_grad()

            output = model(data)

            #print(output, target)


            #print(f"{max(output)}, {min(output)}", f"{max(target)}, {min(output)}")

            if torch.isnan(data).any():
                raise Exception(f"Nan found in data: {data} \r\nor target: {target}")
            if torch.isnan(output).any():
                raise Exception(f"NaN found in output : {output}\r\ndata: {data} \r\ntarget: {target}, {model.classifier.weight}, {model.classifier.bias}")

            loss = loss_fn(output, target)
            
            acc = acc_func(output, target)
            epoch_acc += acc.item()

            loss.backward()
            optimizer.step()
            
        acc = epoch_acc / len(dataloader)
        if use_wandb:
            wandb.log({"train_loss": loss.item(), f"train {acc_label}":acc})
        if (epoch+1) % epoch_log == 0:
            epochs_bar.set_postfix({'last loss':loss.item(), f"last {acc_label}":acc})
            # print(f"Epoch {epoch+1}, loss: {loss.item()}, train {acc_label} : {acc}")

def predict_reg(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        predictions = []
        labels = []
        for data, target in dataloader:
            data = data.to(device)
            output = model(data) # calling model calls forward function
            predictions.extend(output.cpu().numpy())
            labels.extend(target.numpy())
    return np.array(predictions), np.array(labels)

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


def compute_metrics(model, dataloader, device, labels=None, predictions=None):
    metrics_dict = {}
    if labels is None or predictions is None:
        predictions, labels = predict(model, dataloader, device)

    metrics_dict['accuracy'] = accuracy_score(predictions, labels)
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    metrics_dict['roc_auc'] = auc(fpr, tpr)
    metrics_dict['f1_score'] = f1_score(labels, predictions)


    return metrics_dict
