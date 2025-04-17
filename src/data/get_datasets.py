import os

import torchvision.transforms as T

from src.data.datasets import CIFAR10PrepairedConcatenatedDataset, WeightedDataset

def get_cifar10_concatenated(dataset_path=None, dominance_ratio=0.5, is_not_ood=True):
    
    # Przykładowe transformacje, dodaj lepszą
    transform = T.Compose([
        T.ToTensor(),
    ])
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    
    train_dataset = CIFAR10PrepairedConcatenatedDataset(dataset_path, is_train=True, download=True, transform=transform, dominance_ratio=dominance_ratio, is_not_ood=is_not_ood)
    
    test_dataset = CIFAR10PrepairedConcatenatedDataset(dataset_path, is_train=False, download=True, transform=transform, dominance_ratio=dominance_ratio, is_not_ood=is_not_ood)
    
    return train_dataset, test_dataset


def get_cifar10_weighted(dataset_path=None, error_weight=1, failure_indices_path=None):
    
    # Przykładowe transformacje
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    self.train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=download)
    self.test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    
    with open(filename, "rb") as f:
        failure_indices = pickle.load(f)
    
    train_dataset_weighted = WeightedDataset(train_dataset, failure_indices, weight=error_weight)
    
    
    return train_dataset_weighted, test_dataset