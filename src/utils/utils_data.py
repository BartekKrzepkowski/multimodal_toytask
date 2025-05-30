import numpy as np
import pickle
import random
import logging

from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

from src.data.datasets import WeightedDataset
from src.utils.mapping_new import DATASET_NAME_MAP


def prepare_loaders(data_params):
    train_dataset, datasets = DATASET_NAME_MAP[data_params['dataset_name']](**data_params['dataset_params'])
    datasets = (datasets,) if not isinstance(datasets, tuple) else datasets
    
    if (
        'failure_indices_path' in data_params['dataset_params'] 
        and data_params['dataset_params'] ['failure_indices_path'] is not None
    ):
        train_dataset = weight_dataset(
            train_dataset,
            data_params['dataset_params'] ['failure_indices_path'],
            data_params['dataset_params'] ['error_weight'],
            data_params['dataset_params'] ['failure_percentage']
        )
    loaders = {
        'train': DataLoader(train_dataset, shuffle=True, **data_params['loader_params']),
        'test': DataLoader(datasets[0], shuffle=False, **data_params['loader_params'])
    }
    for i, dataset in enumerate(datasets[1:]):
        loaders[f'test_ood{i}'] = DataLoader(dataset, shuffle=False, **data_params['loader_params'])
    
    return loaders


def weight_dataset(dataset, failure_indices_path, error_weight, failure_percentage):
    from sklearn.model_selection import train_test_split
    with open(failure_indices_path, "rb") as f:
        failure_indices = pickle.load(f)

    failure_indices = list(failure_indices)
    logging.info(f"Loaded {len(failure_indices)} failure indices from {failure_indices_path}")

    failure_labels = [dataset[i][1] for i in failure_indices]
    failure_is_noised = [dataset[i][2][0] for i in failure_indices]

    failure_labels_of_0 = [dataset[i][1] for i in failure_indices if dataset[i][1] == 0]
    failure_labels_of_1 = [dataset[i][1] for i in failure_indices if dataset[i][1] == 1]

    # labels = dataset.targets

    n_worst_group_in_error_set = sum([1 for i in range(len(failure_is_noised)) if failure_is_noised[i] is True])
    # worst_group_in_training_set = [i for i in range(len(dataset)) if dataset[i][2] is True]

    n_worst_group_in_error_set_of_0 = sum([1 for i in range(len(failure_is_noised)) if failure_is_noised[i] is True and failure_labels[i] == 0])
    # worst_group_in_training_set_of_0 = [i for i in worst_group_in_training_set if labels[i] == 0]
    n_worst_group_in_error_set_of_1 = sum([1 for i in range(len(failure_is_noised)) if failure_is_noised[i] is True and failure_labels[i] == 1])
    # worst_group_in_training_set_of_1 = [i for i in worst_group_in_training_set if labels[i] == 1]

    logging.info(f"Precision of failure labels: {n_worst_group_in_error_set / len(failure_labels)}")
    # logging.info(f"Recall of failure labels: {n_worst_group_in_error_set / len(worst_group_in_training_set)}")

    logging.info(f"Precision of failure labels of 0: {n_worst_group_in_error_set_of_0 / len(failure_labels_of_0)}") 
    # logging.info(f"Recall of failure labels of 0: {n_worst_group_in_error_set_of_0 / len(worst_group_in_training_set_of_0)}")
    logging.info(f"Precision of failure labels of 1: {n_worst_group_in_error_set_of_1 / len(failure_labels_of_1)}")
    # logging.info(f"Recall of failure labels of 1: {n_worst_group_in_error_set_of_1 / len(worst_group_in_training_set_of_1)}")

    logging.info(f"Distribution of failure labels: {np.unique(failure_labels, return_counts=True)}")
    logging.info(f"Distribution of failure is_noised: {np.unique(failure_is_noised, return_counts=True)}")

    # failure_indices = [i for i in failure_indices if dataset[i][2][0] is False]
    failure_indices = [i for i in failure_indices if dataset[i][2][0] is False]

    # failure_indices_0 = [i for n, i in enumerate(failure_indices) if n < 60]
    # failure_indices_1 = [i for n, i in enumerate(failure_indices) if n < 13]

    # failure_indices = failure_indices_0 + failure_indices_1

    failure_labels = [dataset[i][1] for i in failure_indices]
    failure_is_noised = [dataset[i][2][0] for i in failure_indices]

    logging.info(f"Distribution of failure labels after change: {np.unique(failure_labels, return_counts=True)}")
    logging.info(f"Distribution of failure is_noised after change: {np.unique(failure_is_noised, return_counts=True)}")

    balanced_failure_subset = select_balanced_failure_subset(dataset, failure_indices, percentage=failure_percentage, seed=83)
    logging.info(f"Selected {len(balanced_failure_subset)} failure indices for training")
    logging.info(f"Distribution of selected failure labels: {np.unique([dataset[i][1] for i in balanced_failure_subset], return_counts=True)}")
    logging.info(f"Distribution of selected failure is_noised: {np.unique([dataset[i][2] for i in balanced_failure_subset], return_counts=True)}")
    # dopisz liczenie jaki jest rozkład indeksów do podgrup
    # k = int(len(failure_indices) * failure_percentage)
    # failure_indices = random.sample(failure_indices, k)
    
    dataset = WeightedDataset(dataset, balanced_failure_subset, weight=error_weight)
    return dataset


def save_failure_indices(model, dataset, save_path, device):
    model.eval()
    failure_indices = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            incorrect_mask = (predictions != labels)
            
            # Indeksy błędnych klasyfikacji
            for i, incorrect in enumerate(incorrect_mask):
                if incorrect:
                    failure_indices.append(batch_idx * 64 + i)
                    
    failure_indices = set(failure_indices)
    
    with open(save_path, "wb") as f:
        pickle.dump(failure_indices, f)



def select_balanced_failure_subset(dataset, failure_indices, percentage=0.4, seed=83):
    """
    Wybiera p% błędnych przykładów w sposób zrównoważony klasowo.

    Z każdej klasy bierze równą liczbę przykładów, tak by suma ≈ p% * len(failure_indices)

    Returns: lista indeksów
    """
    random.seed(seed)
    
    # Grupuj błędy wg klasy
    label_to_indices = defaultdict(list)
    for idx in failure_indices:
        label = dataset[idx][1]
        label_to_indices[label].append(idx)

    # Ile przykładów łącznie chcemy?
    total_to_pick = int(len(failure_indices) * percentage)
    num_classes = len(label_to_indices)
    per_class = total_to_pick // num_classes
    per_class = min(per_class, len(label_to_indices[0]), len(label_to_indices[1]))

    selected = []
    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        selected.extend(indices[:per_class])

    return selected
