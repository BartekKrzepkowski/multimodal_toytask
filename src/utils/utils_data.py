import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

from src.data.get_datasets import get_cifar10
from src.utils.utils_visualize import show_and_save_grid


def prepare_loaders(dataset_name, dataset_params, loader_params):
    train_dataset, test_dataset = get_cifar10(**dataset_params)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)
    loaders = {
        'train': train_loader,
        'test': test_loader
    }
    show_and_save_grid(train_dataset, save_path='train_dataset.png')
    show_and_save_grid(train_dataset, save_path='test_dataset.png')
    return loaders


def prepare_loaders_with_weighted_dataset(dataset_name, dataset_params, loader_params):
    train_dataset, test_dataset = get_cifar10(**dataset_params)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)
    loaders = {
        'train': train_loader,
        'test': test_loader
    }
    show_and_save_grid(train_dataset, save_path='train_dataset.png')
    show_and_save_grid(train_dataset, save_path='test_dataset.png')
    return loaders


def get_failure_indices(model, dataset, failure_indices_path):
    model.eval()
    failure_indices = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            incorrect_mask = (predictions != labels)
            
            # Indeksy błędnych klasyfikacji
            for i, incorrect in enumerate(incorrect_mask):
                if incorrect:
                    failure_indices.append(batch_idx * 64 + i)
                    
    failure_indices = set(failure_indices)
    
    with open(failure_indices_path, "wb") as f:
        pickle.dump(failure_indices, f)

    return failure_indices