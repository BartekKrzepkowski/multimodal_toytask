import torch

from src.data.get_datasets import  get_cifar10, get_cifar10_concatenated, get_cats_and_dogs, get_cifar10_cats_and_dogs, get_cats_and_dogs_single, get_cats_and_dogs_blurred

DATASET_NAME_MAP = {
    'cifar10': get_cifar10,
    'get_cifar10_concatenated': get_cifar10_concatenated,
    'cats_and_dogs': get_cats_and_dogs,
    'cifar10_cats_and_dogs': get_cifar10_cats_and_dogs,
    'cats_and_dogs_single': get_cats_and_dogs_single,
    'cats_and_dogs_blurred': get_cats_and_dogs_blurred
}

SCHEDULER_NAME_MAP = {
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'multiplicative': torch.optim.lr_scheduler.MultiplicativeLR,
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}