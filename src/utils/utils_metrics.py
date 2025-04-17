import torch

def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    acc = correct / y_true.size(0)
    return acc

def mean(lst):
    return sum(lst) / len(lst)
    