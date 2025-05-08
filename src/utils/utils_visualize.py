import logging
import os
from datetime import datetime

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

def show_and_save_grid(dataset, num_images=16, cols=4, save_path="dataset_grid.png"):
    # wybieramy losowe indeksy obrazów
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    images = [dataset[i][0] for i in indices]
    labels = [dataset[i][1] for i in indices]

    # tworzymy siatkę
    grid = make_grid(images, nrow=cols, padding=2)

    # konwersja do formatu numpy (dla matplotlib)
    npimg = grid.numpy()
    plt.figure(figsize=(cols*3, (num_images//cols)*3))
    plt.axis('off')
    plt.title('Przykładowe obrazy z datasetu')

    # matplotlib oczekuje formatu (wys, szer, kanały)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    # zapisujemy obraz
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Zapisano obraz jako {save_path}")

    # wyświetlamy
    plt.show()
    
    
    
    
def matplotlib_scatters_training(config, epochs_metrics):
    # Tworzenie wykresu metryk: strata i dokładność
    plt.figure(figsize=(10, 4))
    
    # Wykres strat
    plt.subplot(1, 2, 1)
    for metric_name in epochs_metrics:
        if 'loss' in metric_name:
            plt.plot(
                range(1, config.n_epochs + 1),
                epochs_metrics[metric_name],
                label=metric_name
            )
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True)
    plt.legend()
    
    # Wykres dokładności
    plt.subplot(1, 2, 2)
    for metric_name in epochs_metrics:
        if 'acc' in metric_name:
            plt.plot(
                range(1, config.n_epochs + 1),
                epochs_metrics[metric_name],
                label=metric_name
            )
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.grid(True)
    plt.legend()
    
    # Zapisywanie wykresu do folderu 'results'
    if not os.path.exists("results"):
        os.makedirs("results")
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(
        os.path.join("results", f"metrics_{config.postfix_title}_{timestamp}.pdf"),
        bbox_inches='tight'
    )
    
    
def log_to_console(epochs_metrics):
    last_vals = {
        key: vals[-1] if len(vals) > 0 else 0.0
        for key, vals in epochs_metrics.items()
        if key != 'epoch'
    }

    parts = [
        f"{key}: {last_vals[key]:.4f}"
        for key in sorted(last_vals)
    ]
    metrics_str = ", ".join(parts)
    logging.info(f"Epoka {epochs_metrics['epoch']}: {metrics_str}")