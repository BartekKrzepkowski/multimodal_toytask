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
    
    
    
    
def matplotlib_scatters_training(n_epochs, train_losses, test_losses, train_acc, test_acc):
    # Tworzenie wykresu metryk: strata i dokładność
    plt.figure(figsize=(10, 4))
    
    # Wykres strat
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), train_losses, label="Strata treningowa")
    plt.plot(range(1, n_epochs + 1), test_losses, label="Strata testowa")
    plt.xlabel("Epoka")
    plt.ylabel("Strata (Cross Entropy)")
    plt.legend()
    
    # Wykres dokładności
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), train_acc, label="Dokładność treningowa")
    plt.plot(range(1, n_epochs + 1), test_acc, label="Dokładność testowa")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność")
    plt.legend()
    
    # Zapisywanie wykresu do folderu 'results'
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig(os.path.join("results", f"metrics_{config.postfix_title}.pdf"), bbox_inches='tight')
    
