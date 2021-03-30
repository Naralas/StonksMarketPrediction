import torch
import torch.nn as nn

def train(dataloader, model, n_epochs, optimizer, loss_fn, device):
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

        print(f"Epoch {epoch+1}, loss: {loss.item()}, accuracy : {epoch_acc / len(dataloader):.2f}")

def binary_acc(output, target):
    # https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
    output_tag = torch.round(torch.sigmoid(output))

    correct_results_sum = (output_tag == target).sum().float()
    acc = correct_results_sum/target.shape[0]
    acc = torch.round(acc * 100)
    
    return acc