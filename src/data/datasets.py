import logging
import pickle
import random
import os
from collections import defaultdict
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import datasets


class DatasetClassRemapped(Dataset):
    def __init__(self, base_dataset, class_mapping):
        self.base_dataset = base_dataset
        self.class_mapping = class_mapping
        self.transform = getattr(base_dataset, "transform", None)
        self.targets = base_dataset.targets

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.class_mapping[label]

    def __len__(self):
        return len(self.base_dataset)
    
    
class RemappedSubsetDataset(Dataset):
    def __init__(self, base_dataset, indices, class_mapping, transform=None):
        """
        :param base_dataset: oryginalny dataset (np. ImageFolder, OxfordIIITPet)
        :param indices: lista indeksów (podzbiór)
        :param class_mapping: słownik mapujący oryginalne etykiety -> nowe etykiety
        :param transform: opcjonalna transformacja obrazu
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.class_mapping = class_mapping
        self.transform = transform or getattr(base_dataset, 'transform', None)
        self.targets = [class_mapping[label] for label in np.array(base_dataset.targets)[np.array(indices)]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        label = self.class_mapping[label]
        if self.transform:
            image = self.transform(image)
        return image, label


class CIFAR10PrepairedConcatenatedDataset(Dataset):
    def __init__(self, dataset_path, is_train=True, download=False, transform=None, dominance_ratio=0.7, is_not_ood=True, indices_file=None):
        dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
        self.cifar_dataset = datasets.CIFAR10(root=dataset_path, train=is_train, download=download)
        self.transform = transform
        self.is_not_ood = is_not_ood
        self.is_train = is_train
        
        if indices_file is None:
            indices_file = "data/train_indices.pkl" if is_train else ("data/test_indices.pkl" if is_not_ood else "data/test_ood_indices.pkl")

        # Jeśli plik istnieje, wczytaj zapisane indeksy
        if os.path.exists(indices_file):
            with open(indices_file, "rb") as f:
                self.paired_indices = pickle.load(f)
            print(f"Wczytano indeksy z {indices_file}")
        else:
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
                    # przypisanie klasy zgodnie z dominance_ratio
                    assigned_label = left_class if random.random() < dominance_ratio else right_class
                    self.paired_indices.append({
                        'left_idx': left_indices[i],
                        'right_idx': right_indices[i],
                        'assigned_label': assigned_label,
                        'ood_label': right_class if assigned_label == left_class else left_class
                    })
            # Zapis słownika do pliku
            with open(indices_file, "wb") as f:
                pickle.dump(self.paired_indices, f)
            print(f"Zapisano indeksy do {indices_file}")

    def __len__(self):
        return len(self.paired_indices)

    def __getitem__(self, idx):
        pair_info = self.paired_indices[idx]

        img_left, _ = self.cifar_dataset[pair_info['left_idx']]
        img_right, _ = self.cifar_dataset[pair_info['right_idx']]
        
        if self.is_train:
            if random.random() < 0.5:
                img_left = img_left.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img_right = img_right.transpose(Image.FLIP_LEFT_RIGHT)

        concatenated_img = Image.fromarray(
            np.hstack((np.array(img_left), np.array(img_right)))
        )


        assigned_label = pair_info['assigned_label'] if (self.is_not_ood or self.is_train) else pair_info['ood_label']
        if self.transform:
            concatenated_img = self.transform(concatenated_img)

        return concatenated_img, assigned_label
    
    
# JTT DataLoader (z wagami)
class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, failure_indices, weight=20):
        logging.info(f"Using weight {weight} for {len(failure_indices)} failure indices.")
        self.dataset = original_dataset
        self.failure_indices = failure_indices
        self.weight = weight

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, is_corrupted = self.dataset[idx]
        sample_weight = self.weight if idx in self.failure_indices else 1
        return img, label, [is_corrupted[0], sample_weight] # ponieważ is_corrupted to lista długości 1


class PairedSameClassDatasetWithOOD(Dataset):
    def __init__(self, dataset_a, dataset_b, dataset_c=None,
                 is_train=True, is_not_ood=True, indices_file=None):
        """
        dataset_a : PyTorch Dataset (np. CIFAR-10)
        dataset_b : PyTorch Dataset (np. Stanford Dogs)
        dataset_c : PyTorch Dataset (np. Oxford Pets) - używany przy OOD
        class_mapping_a, class_mapping_b, class_mapping_c : mapowania klas
        is_train : czy jesteśmy w fazie treningu
        transform : transformacje do nałożenia po konkatenacji
        is_not_ood : czy to zbiór normalny czy OOD
        indices_file : ścieżka do pliku z zapisanymi indeksami
        """

        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.dataset_c = dataset_c
        self.is_train = is_train
        self.is_not_ood = is_not_ood
        self.resize_transform = T.Compose([
            T.Resize((32, 32), interpolation=T.InterpolationMode.BICUBIC)
        ])
        
        self.full_transform = T.Compose([
            # T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # T.RandomCrop((96, 96)),
            T.RandomAffine(degrees=10, translate=(1/8, 1/8)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        if indices_file is None:
            indices_file = "data/train_indices_3.pkl" if is_train else ("data/test_indices_3.pkl" if is_not_ood else "data/test_ood_indices_3.pkl")

        if os.path.exists(indices_file):
            with open(indices_file, "rb") as f:
                self.paired_indices = pickle.load(f)
            print(f"Wczytano indeksy z {indices_file}")
        else:
            # Mapowanie klasa -> lista indeksów
            self.class_to_indices_a = {}
            self.class_to_indices_b = {}
            self.class_to_indices_c = {}

            for idx, label in enumerate(self.dataset_a.targets):
                self.class_to_indices_a.setdefault(label, []).append(idx)

            for idx, label in enumerate(self.dataset_b.targets):
                self.class_to_indices_b.setdefault(label, []).append(idx)

            for idx, label in enumerate(self.dataset_c.targets):
                self.class_to_indices_c.setdefault(label, []).append(idx)

            # Znajdujemy wspólne klasy we wszystkich trzech datasetach
            self.shared_classes = list(set(self.class_to_indices_a.keys()) &
                                       set(self.class_to_indices_b.keys()) &
                                       set(self.class_to_indices_c.keys()))
            if not self.shared_classes:
                raise ValueError("Brak wspólnych klas między datasetami A, B i C!")

            # Tworzymy pary
            self.paired_indices = []
            for shared_class in self.shared_classes:
                left_indices = self.class_to_indices_a[shared_class].copy()
                middle_indices = self.class_to_indices_b[shared_class].copy()
                right_indices = self.class_to_indices_c[shared_class].copy()

                random.shuffle(middle_indices)
                random.shuffle(right_indices)

                pair_count = min(len(left_indices), len(middle_indices), len(right_indices))

                for i in range(pair_count):
                    self.paired_indices.append({
                        'left_idx': left_indices[i],     # dataset_a
                        'middle_idx': middle_indices[i], # dataset_b
                        'right_idx': right_indices[i],   # dataset_c (OOD)
                        'assigned_label': shared_class,
                    })

            # Zapisujemy pary
            with open(indices_file, "wb") as f:
                pickle.dump(self.paired_indices, f)
            print(f"Zapisano indeksy do {indices_file}")

    def __len__(self):
        return len(self.paired_indices)

    def __getitem__(self, idx):
        pair_info = self.paired_indices[idx]

        img_left, _ = self.dataset_a[pair_info['left_idx']]
        img_middle, _ = self.dataset_b[pair_info['middle_idx']]
        img_right, _ = self.dataset_c[pair_info['right_idx']]  # używany w OOD

        # W fazie treningu i standardowego testu używamy A + B
        # W fazie OOD używamy A + C
        if self.is_not_ood or self.is_train:
            second_img = img_middle
        else:
            second_img = img_right

        if self.is_train:
            if random.random() < 0.5:
                img_left = img_left.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                second_img = second_img.transpose(Image.FLIP_LEFT_RIGHT)

        
        img_left = self.resize_transform(img_left)
        second_img = self.resize_transform(second_img)
        
        # Konkatenacja pozioma (left + middle lub left + right)
        concatenated_img = Image.fromarray(
            np.hstack((np.array(img_left), np.array(second_img)))
        )

        concatenated_img = self.full_transform(concatenated_img)

        label = pair_info['assigned_label']

        return concatenated_img, label


class ConcatNoisyDataset(Dataset):
    def __init__(
        self,
        dataset1: Dataset,
        dataset2: Dataset,
        phase: str = 'train',
        blur_prob: float = 0.5,   # prawdopodobieństwo rozmycia drugiego obrazu
        indices_file: str = None,
        seed: int = 42,
        blur_opposite: bool = False,    # czy rozmywać obraz przeciwny
        is_blur: bool = True    # czy używać rozmycia (True) czy szumu (False)
    ):
        """
        Łączy obrazy tej samej klasy z dwóch datasetów.
        Z losowym (ale deterministycznym) szumem na drugim obrazie.
        Pary indeksów + flaga szumu zapisuje / wczytuje przez pickle.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.targets = dataset1.targets
        self.blur_prob = blur_prob
        self.indices_file = indices_file
        self.is_train = (phase == 'train')
        self.blur_opposite = blur_opposite
        self.is_blur = is_blur
        
        self.resize_transform = T.Resize((32, 32), interpolation=T.InterpolationMode.BICUBIC)
        self.horizontal_flip_transform = T.RandomHorizontalFlip(p=0.5)
        self.transform_blurred = T.Compose([
            T.Resize((8, 8), interpolation=T.InterpolationMode.BILINEAR, antialias=None),
            T.Resize((32, 32), interpolation=T.InterpolationMode.BILINEAR, antialias=None)
        ])
        self.full_transform = T.Compose([
            # T.RandomHorizontalFlip(p=0.5),
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # T.RandomCrop((96, 96)),
            T.RandomAffine(degrees=10, translate=(1/8, 1/8)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if indices_file is None:
            indices_file = f"data/{phase}_indices_blur_prob_{blur_prob}.pkl"

        # 1) Ustawienie ziarna dla powtarzalności
        if seed is not None:
            random.seed(seed)

        # 2) Jeśli istnieje plik z parami → wczytaj je
        if os.path.exists(indices_file):
            logging.info(f"Wczytano pary z {indices_file}")
            with open(indices_file, 'rb') as f:
                self.paired_indices = pickle.load(f)
        else:
            logging.info(f"Nie znaleziono pliku {indices_file}. Tworzenie nowych par...")
            # 3) Zbuduj mapę klasy → indeksy w dataset2
            class_to_idxs1 = defaultdict(list)
            class_to_idxs2 = defaultdict(list)
            for idx1, label1 in enumerate(self.dataset1.targets):
                class_to_idxs1[label1].append(idx1)
            for idx2, label2 in enumerate(self.dataset2.targets):
                class_to_idxs2[label2].append(idx2)
            
                
                
            # Znajdujemy wspólne klasy we wszystkich trzech datasetach
            self.shared_classes = list(set(class_to_idxs1.keys()) &
                                       set(class_to_idxs2.keys()))
            if not self.shared_classes:
                raise ValueError("Brak wspólnych klas między datasetami 1 i 2!")
            
            # Tworzymy pary
            self.paired_indices = []
            for shared_class in self.shared_classes:
                left_indices = class_to_idxs1[shared_class].copy()
                right_indices = class_to_idxs2[shared_class].copy()

                random.shuffle(right_indices)

                pair_count = min(len(left_indices), len(right_indices))

                for i in range(pair_count):
                    self.paired_indices.append({
                        'left_idx': left_indices[i],     # dataset_a
                        'right_idx': right_indices[i],   # dataset_c (OOD)
                        'apply_noise': (random.random() < self.blur_prob),
                    })

            # 5) Zapisz pary na dysku, jeśli podano ścieżkę
            if indices_file:
                logging.info(f"Zapisano pary do {indices_file}")
                os.makedirs(os.path.dirname(indices_file), exist_ok=True)
                with open(indices_file, 'wb') as f:
                    pickle.dump(self.paired_indices, f)

    def __len__(self):
        return len(self.paired_indices)

    def __getitem__(self, idx):
        pair_info = self.paired_indices[idx]
        img_left, label = self.dataset1[pair_info['left_idx']]
        img_right, _   = self.dataset2[pair_info['right_idx']]

        if self.is_train:
            img_left = self.horizontal_flip_transform(img_left)
            img_right = self.horizontal_flip_transform(img_right)
        
        img_right = self.resize_transform(img_right)
        if pair_info['apply_noise']:
            if self.blur_opposite:
                img_left = self.transform_blurred(img_left) if self.is_blur else np.random.randint(0, 256, size=(img_left.size[0], img_left.size[1], 3), dtype=np.uint8)
            else:
                img_right = self.transform_blurred(img_right) if self.is_blur else np.random.randint(0, 256, size=(img_left.size[0], img_left.size[1], 3), dtype=np.uint8)
            
        concatenated_img = Image.fromarray(
            np.hstack((np.array(img_left), np.array(img_right)))
        )
        concatenated_img = self.full_transform(concatenated_img)
        return concatenated_img, label, [pair_info['apply_noise']]