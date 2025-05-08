import os
import random

import numpy as np
import torch

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint zapisano w {filename}")
    
    
def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint załadowany z epoki {epoch}, loss: {loss:.4f}")
    return epoch, loss


def load_model(model, path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
    return model


def save_training_artefacts(config, epochs_metrics, save_path):
    # Zapisywanie konfiguracji, metryk treningowych oraz metryk wartości osobliwych
    to_save = {
        'config': config,
        'metrics': epochs_metrics,
    }
    torch.save(to_save, os.path.join(save_path, "info.pth"))