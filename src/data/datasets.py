import random
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import datasets

class CIFAR10ModalDominanceDataset(Dataset):
    def __init__(self, dataset_path, is_train=True, download=False, transform=None, dominance_ratio=0.7, is_not_ood=False):
        dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
        self.cifar_dataset = datasets.CIFAR10(root=dataset_path, train=is_train, download=download)
        self.transform = transform
        self.dominance_ratio = dominance_ratio
        self.is_not_ood = is_not_ood
        self.is_train = is_train
        
        # Definiujemy pary klas (lewa - prawa)
        self.class_pairs = [(0,9), (1,8), (2,7), (3,6), (4,5)]
        
        # Indeksy obrazów dla każdej klasy
        self.class_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.cifar_dataset):
            self.class_to_indices[label].append(idx)
        
        # Rozmiar datasetu
        self.length_per_pair = min(
            [min(len(self.class_to_indices[a]), len(self.class_to_indices[b])) 
             for a, b in self.class_pairs]
        )

    def __len__(self):
        return self.length_per_pair * len(self.class_pairs)

    def __getitem__(self, idx):
        pair_idx = idx % len(self.class_pairs)
        class_left, class_right = self.class_pairs[pair_idx]
        
        # Losowy wybór obrazów z każdej klasy
        img_idx_left = random.choice(self.class_to_indices[class_left])
        img_idx_right = random.choice(self.class_to_indices[class_right])
        
        img_left, _ = self.cifar_dataset[img_idx_left]
        img_right, _ = self.cifar_dataset[img_idx_right]
        
        # Konkatenacja pozioma obrazów
        concatenated_img = Image.fromarray(
            np.hstack((np.array(img_left), np.array(img_right)))
        )
        
        # Ustalenie klasy dominującej zgodnie z zadanymi proporcjami
        if random.random() < self.dominance_ratio:
            assigned_label = class_left if (self.is_not_ood or self.is_train) else class_right # 70% przypadków
        else:
            assigned_label = class_right if (self.is_not_ood or self.is_train) else class_left  # 30% przypadków
        
        if self.transform:
            concatenated_img = self.transform(concatenated_img)
        
        return concatenated_img, assigned_label


class CIFAR10PrepairedConcatenatedDataset(Dataset):
    def __init__(self, dataset_path, is_train=True, download=False, transform=None, dominance_ratio=0.7, is_not_ood=False):
        dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
        self.cifar_dataset = datasets.CIFAR10(root=dataset_path, train=is_train, download=download)
        self.transform = transform
        self.dominance_ratio = dominance_ratio
        self.is_not_ood = is_not_ood
        self.is_train = is_train

        # pary klas do konkatenacji
        self.class_pairs = [(0,9), (1,8), (2,7), (3,6), (4,5)]
        
        # indeksy dla każdej klasy
        self.class_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.cifar_dataset):
            self.class_to_indices[label].append(idx)
        
        # Przygotowanie par indeksów bez powtórzeń
        self.paired_indices = []
        for left_class, right_class in self.class_pairs:
            left_indices = self.class_to_indices[left_class].copy()
            right_indices = self.class_to_indices[right_class].copy()

            random.shuffle(left_indices)
            random.shuffle(right_indices)

            pair_count = min(len(left_indices), len(right_indices))

            # Tworzymy pary
            for i in range(pair_count):
                self.paired_indices.append({
                    'left_idx': left_indices[i],
                    'right_idx': right_indices[i],
                    'left_class': left_class,
                    'right_class': right_class
                })

    def __len__(self):
        return len(self.paired_indices)

    def __getitem__(self, idx):
        pair_info = self.paired_indices[idx]

        img_left, _ = self.cifar_dataset[pair_info['left_idx']]
        img_right, _ = self.cifar_dataset[pair_info['right_idx']]

        concatenated_img = Image.fromarray(
            np.hstack((np.array(img_left), np.array(img_right)))
        )

        # przypisanie klasy zgodnie z dominance_ratio
        if random.random() < self.dominance_ratio:
            assigned_label = pair_info['left_class'] if (self.is_not_ood or self.is_train) else pair_info['right_class'] # p*100% przypadków
        else:
            assigned_label = pair_info['right_class'] if (self.is_not_ood or self.is_train) else pair_info['left_class'] # (1-p)*100% przypadków

        if self.transform:
            concatenated_img = self.transform(concatenated_img)

        return concatenated_img, assigned_label
    
    
# JTT DataLoader (z wagami)
class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, failure_indices, weight=3):
        self.dataset = original_dataset
        self.failure_indices = failure_indices
        self.weight = weight  # np. 3x większa waga dla błędów

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        sample_weight = self.weight if idx in self.failure_indices else 1
        return img, label, sample_weight
