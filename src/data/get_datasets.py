import logging
import pickle
import os

import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split

from src.data.datasets import CIFAR10PrepairedConcatenatedDataset, DatasetClassRemapped, PairedSameClassDatasetWithOOD, ConcatNoisyDataset, RemappedSubsetDataset

def get_cifar10_concatenated(dataset_path=None, dominance_ratio=0.5, **kwargs):
    
    # Przykładowe transformacje, dodaj lepszą
    # transform = T.Compose([
    #     T.ToTensor(),
    # ])
    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    normalize = T.Normalize(mean, std)
    transform_train = T.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(1/8, 1/8)),
        T.ToTensor(),
        normalize,
        # transforms.RandomErasing(p=0.05),
    ])
    transform_eval = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    
    train_dataset = CIFAR10PrepairedConcatenatedDataset(dataset_path, is_train=True, download=True, transform=transform_train, dominance_ratio=dominance_ratio,
                                                        indices_file=kwargs['train_indices_file'])
    
    test_dataset = CIFAR10PrepairedConcatenatedDataset(dataset_path, is_train=False, download=True, transform=transform_eval, dominance_ratio=dominance_ratio,
                                                       is_not_ood=True, indices_file=kwargs['test_indices_file'])
    test_ood_dataset = CIFAR10PrepairedConcatenatedDataset(dataset_path, is_train=False, download=True, transform=transform_eval, dominance_ratio=dominance_ratio,
                                                           is_not_ood=False, indices_file=kwargs['test_ood_indices_file'])
    
    return train_dataset, test_dataset, test_ood_dataset


# 1. CIFAR-10
def get_cifar10_cats_and_dogs(dataset_path=None, **kwargs):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    if kwargs['use_transform']:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
        normalize = T.Normalize(mean, std)
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(1/8, 1/8)),
            T.ToTensor(),
            normalize,
            # transforms.RandomErasing(p=0.05),
        ])
        transform_eval = T.Compose([
            T.ToTensor(),
            normalize,
        ])
    else:
        transform_train = None
        transform_eval = None
    
    train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_eval)
    
    
    selected_indices_train = [idx for idx, (_, label) in enumerate(train_dataset) if label in [3, 5]]
    selected_indices_test = [idx for idx, (_, label) in enumerate(test_dataset) if label in [3, 5]]
    
    # Mapowanie klas CIFAR-10
    # 3: cat -> 0, 5: dog -> 1
    class_mapping_cifar = {3: 0, 5: 1}
    
    selected_indices_train = limit_indices(selected_indices_train, 5000, train_dataset, class_mapping_cifar)
    selected_indices_test = limit_indices(selected_indices_test, 1000, test_dataset, class_mapping_cifar)
    
    # train_dataset = Subset(train_dataset, selected_indices_train)
    # test_dataset = Subset(test_dataset, selected_indices_test)
    
    train_dataset = RemappedSubsetDataset(train_dataset, selected_indices_train, class_mapping_cifar)
    test_dataset = RemappedSubsetDataset(test_dataset, selected_indices_test, class_mapping_cifar)
    
    return train_dataset, test_dataset


# 2. Kaggle Cats and Dogs
def get_kaggle_cvsd(dataset_path=None, indices_file_cvsd=None, **kwargs):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CVSD_PATH']
    dataset = datasets.ImageFolder(root=dataset_path)
    
    if indices_file_cvsd is None:
        indices_file_cvsd = 'data/cvsd_split_indices.pkl'

    # Jeśli plik z indeksami istnieje – wczytaj
    if os.path.exists(indices_file_cvsd):
        with open(indices_file_cvsd, 'rb') as f:
            indices = pickle.load(f)
        print(f"✅ Wczytano indeksy z {indices_file_cvsd}")
    else:
        all_indices = list(range(len(dataset)))
        all_labels = [dataset.samples[i][1] for i in all_indices]
        # Podział 80/20 z ustalonym ziarnem
        train_idx, test_idx = train_test_split(
            all_indices,
            train_size=0.8,
            stratify=all_labels,
            random_state=42
        )
        
        indices = {'train': train_idx, 'test': test_idx}
        os.makedirs(os.path.dirname(indices_file_cvsd), exist_ok=True)
        with open(indices_file_cvsd, 'wb') as f:
            pickle.dump(indices, f)
        print(f"✅ Zapisano indeksy do {indices_file_cvsd}")
        
        
    class_mapping = {0: 0, 1: 1}
    train_idx = limit_indices(indices['train'], 5000, dataset, class_mapping)
    test_idx = limit_indices(indices['test'], 1000, dataset, class_mapping)

    # Tworzenie podzbiorów z indeksów
    # train_dataset = Subset(dataset, train_idx)
    # test_dataset = Subset(dataset, test_idx)
    
    train_dataset = RemappedSubsetDataset(dataset, train_idx, class_mapping)
    test_dataset = RemappedSubsetDataset(dataset, test_idx, class_mapping)
    
    logging.info(f"Zbiór treningowy cvsd: {len(train_dataset)} próbek")
    logging.info(f"Zbiór testowy cvsd: {len(test_dataset)} próbek")

    return train_dataset, test_dataset


# 3. Oxford Pets
def get_oxford_pets_cats_and_dogs(dataset_path=None, split_ratio=0.8, seed=42, **kwargs):
    dataset_path = dataset_path or os.environ['OXFORD_PETS_PATH']

    # Wczytanie pełnego datasetu (wszystkie rasy)
    full_dataset = datasets.OxfordIIITPet(root=dataset_path, download=True, target_types="category")

    # Ustalmy klasy kotów i psów
    # class_to_species = full_dataset._class_to_species  # dict: class_name -> species
    # label_to_species = {v['label']: v['species'] for v in class_to_species.values()}  # int_label -> 'cat'/'dog'
    
    
    # Etap 1: Zdefiniuj mapowanie klas -> kot/pies
    cat_labels = list(range(0, 12))      # klasy 0–11 = koty
    dog_labels = list(range(12, 37))     # klasy 12–36 = psy

    class_mapping = {label: 0 for label in cat_labels}
    class_mapping.update({label: 1 for label in dog_labels})

    # Etap 2: Wybierz indeksy tylko dla kotów i psów
    selected_indices = []
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        if label in class_mapping:
            selected_indices.append(idx)
        
    test_idx = limit_indices(selected_indices, 1000, full_dataset, class_mapping)
    
    # logging.info(f"Zbiór testowy oxford: {len(test_idx)} próbek")
    
    # test_dataset = Subset(full_dataset, test_idx)
    test_dataset = RemappedSubsetDataset(full_dataset, test_idx, class_mapping)
    
    logging.info(f"Zbiór testowy oxford: {len(test_dataset)} próbek.")

    return test_dataset


def get_cats_and_dogs(**kwargs):
    train_dataset_cifar10, test_dataset_cifar10 = get_cifar10_cats_and_dogs(**kwargs)
    logging.info("Cifar10 cats and dogs loaded.")
    train_dataset_cvsd, test_dataset_cvsd = get_kaggle_cvsd(**kwargs)
    logging.info("CVSD cats and dogs loaded.")
    test_dataset_oxford_pets = get_oxford_pets_cats_and_dogs(**kwargs)
    logging.info("Oxford pets cats and dogs loaded.")
    
    train_dataset = PairedSameClassDatasetWithOOD(train_dataset_cifar10, train_dataset_cvsd, train_dataset_cvsd,
                                                  is_train=True, is_not_ood=True)
    
    test_dataset = PairedSameClassDatasetWithOOD(test_dataset_cifar10, test_dataset_cvsd, test_dataset_oxford_pets,
                                                  is_train=False, is_not_ood=True)
    
    test_ood_dataset = PairedSameClassDatasetWithOOD(test_dataset_cifar10, test_dataset_cvsd, test_dataset_oxford_pets,
                                                  is_train=False, is_not_ood=False)
    
    return train_dataset, (test_dataset, test_ood_dataset)


def get_cats_and_dogs_single(**kwargs):
    import copy
    train_dataset_cifar10, test_dataset_cifar10 = get_cifar10_cats_and_dogs(**kwargs)
    _, test_dataset_cvsd = get_kaggle_cvsd(**kwargs)
    test_dataset_oxford_pets = get_oxford_pets_cats_and_dogs(**kwargs)
    
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    normalize = T.Normalize(mean, std)
    transform_train = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(1/8, 1/8)),
        T.ToTensor(),
        normalize,
        # transforms.RandomErasing(p=0.05),
    ])
    resize_transform = lambda mean, std: T.Compose([
            T.Resize((32, 32), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std),
    ])
    _, test_dataset_cifar102 = get_cifar10_cats_and_dogs(use_transform=False)
    test_dataset_cifar10_cvsd = PairedSameClassDatasetWithOOD(test_dataset_cifar102, copy.deepcopy(test_dataset_cvsd), copy.deepcopy(test_dataset_oxford_pets),
                                                  is_train=False, is_not_ood=True)
    
    test_dataset_cifar10_oxford_pets = PairedSameClassDatasetWithOOD(test_dataset_cifar102, copy.deepcopy(test_dataset_cvsd), copy.deepcopy(test_dataset_oxford_pets),
                                                  is_train=False, is_not_ood=False)
    
    train_dataset_cifar10.transform = transform_train
    test_dataset_cifar10.transform = resize_transform(mean, std)
    test_dataset_cvsd.transform = resize_transform(mean, std)
    test_dataset_oxford_pets.transform = resize_transform(mean, std)
    
    
    
    return train_dataset_cifar10, (test_dataset_cifar10, test_dataset_cvsd, test_dataset_oxford_pets, test_dataset_cifar10_cvsd, test_dataset_cifar10_oxford_pets)


def get_cifar10(dataset_path=None, **kwargs):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    
    # transformacje
    # mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    normalize = T.Normalize(mean, std)
    transform_train = T.Compose([
        # T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(1/8, 1/8)),
        T.ToTensor(),
        normalize,
        # transforms.RandomErasing(p=0.05),
    ])
    transform_eval = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    # Wczytanie zbioru CIFAR-10
    train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_eval)
    return train_dataset, test_dataset


from collections import defaultdict
def limit_indices(indices, limit_per_class, full_dataset, class_mapping):
    labels = full_dataset.targets
    limited = []
    c0 = c1 = 0
    for idx in indices:
        bl = class_mapping[labels[idx]]      # bardzo szybki dostęp
        if bl == 0 and c0 < limit_per_class:
            limited.append(idx); c0 += 1
        elif bl == 1 and c1 < limit_per_class:
            limited.append(idx); c1 += 1
        if c0 >= limit_per_class and c1 >= limit_per_class:
            break
    return limited


def get_cats_and_dogs_blurred(**kwargs):
    train_dataset_cifar10, test_dataset_cifar10 = get_cifar10_cats_and_dogs(**kwargs)
    logging.info("Cifar10 cats and dogs loaded.")
    train_dataset_cvsd, test_dataset_cvsd = get_kaggle_cvsd(**kwargs)
    logging.info("CVSD cats and dogs loaded.")
    
    train_dataset = ConcatNoisyDataset(train_dataset_cifar10, train_dataset_cvsd,
                                                  phase='train', blur_prob=kwargs['blur_prob'], is_blur=kwargs['is_blur'])
    
    test_dataset = ConcatNoisyDataset(test_dataset_cifar10, test_dataset_cvsd,
                                                  phase='test', blur_prob=kwargs['blur_prob'], is_blur=kwargs['is_blur'])
    
    test_ood_dataset = ConcatNoisyDataset(test_dataset_cifar10, test_dataset_cvsd,
                                                  phase=f'test_blur_prob={0}', blur_prob=0.0, is_blur=kwargs['is_blur']) #never blur
    
    test_ood2_dataset = ConcatNoisyDataset(test_dataset_cifar10, test_dataset_cvsd,
                                                  phase=f'test_blur_prob={1}', blur_prob=1.0, is_blur=kwargs['is_blur']) #always blur
    
    test_ood3_dataset = ConcatNoisyDataset(test_dataset_cifar10, test_dataset_cvsd,
                                                  phase=f'test_blur_prob={1}_opposite', blur_prob=1.0, blur_opposite=True, is_blur=kwargs['is_blur']) #always blur, but another side
    
    return train_dataset, (test_dataset, test_ood_dataset, test_ood2_dataset, test_ood3_dataset)

