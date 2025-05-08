from datetime import datetime
import numpy as np
import pickle
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

from src.data.datasets import WeightedDataset
from src.utils.utils_visualize import show_and_save_grid
from src.utils.mapping_new import DATASET_NAME_MAP


def prepare_loaders(dataset_name, dataset_params, loader_params):
    logging.info(1)
    train_dataset, datasets = DATASET_NAME_MAP[dataset_name](**dataset_params)
    logging.info(2)
    datasets = (datasets,) if not isinstance(datasets, tuple) else datasets
    
    if (
        'failure_indices_path' in dataset_params 
        and dataset_params['failure_indices_path'] is not None
    ):
        train_dataset = weight_dataset(
            train_dataset,
            dataset_params['failure_indices_path'],
            dataset_params['error_weight']
        )
    logging.info(3)
    loaders = {
        'train': DataLoader(train_dataset, shuffle=True, **loader_params),
        'test': DataLoader(datasets[0], shuffle=False, **loader_params)
    }
    for i, dataset in enumerate(datasets[1:]):
        loaders[f'test_ood_{i}'] = DataLoader(dataset, shuffle=False, **loader_params)
    
    show_and_save_grid(train_dataset, save_path='train_dataset.png')
    for i, dataset in enumerate(datasets):
        show_and_save_grid(dataset, save_path=f'test_dataset_{i}.png')
    return loaders


def weight_dataset(dataset, failure_indices_path, error_weight):
    with open(failure_indices_path, "rb") as f:
        failure_indices = pickle.load(f)
    
    dataset = WeightedDataset(dataset, failure_indices, weight=error_weight)
    return dataset


def get_failure_indices(model, dataset, failure_indices_path, device):
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
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    failure_indices_path = f"{failure_indices_path}_{timestamp}"
    
    with open(failure_indices_path, "wb") as f:
        pickle.dump(failure_indices, f)

    return failure_indices
