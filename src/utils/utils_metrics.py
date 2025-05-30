import torch

def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum()
    acc = correct / y_true.size(0)
    return acc.item()

def mean(lst):
    return sum(lst) / len(lst)
    