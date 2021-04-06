import torch
import torch.nn as nn
import wandb 
import numpy as np

def train(dataloader, model, n_epochs, optimizer, loss_fn, device, project_label=None):
    for params in optimizer.param_groups:
        lr = params['lr']
    #wandb.init(project=f"Project")
    #config = wandb.config
    #config.learning_rate = lr
    model.train()
    
    #wandb.watch(model)
    for epoch in range(n_epochs):
        epoch_acc = 0
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(data)
            #sum_non_zero += np.count_nonzero(output.cpu() > 0.5)
            
            loss = loss_fn(output, target.unsqueeze(1))
            acc = binary_acc(output, target.unsqueeze(1))
            epoch_acc += acc.item()
            
            loss.backward()
            optimizer.step()

        acc = epoch_acc / len(dataloader)
        #print(f"Non zero : {sum_non_zero}, {sum_non_zero / len(dataloader)}")
        #wandb.log({"loss": loss.item(), "accuracy":acc})
        print(f"Epoch {epoch+1}, loss: {loss.item()}, accuracy : {acc:.2f}")

def predict(dataloader, model, device):
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