import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.utils.utils_data import prepare_loaders
from src.utils.utils_model import prepare_model
from src.utils.utils_optim import prepare_optim_and_scheduler
from src.utils.utils_metrics import accuracy, mean
from src.utils.utils_visualize import matplotlib_scatters_training
from src.utils.utils_general import set_seed,save_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

# Funkcja trenująca model
def train_model(loaders, model, optimizer, criterion, config):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(config.n_epochs):
        train_losses_per_epoch = []
        test_losses_per_epoch = []
        train_accs_per_epoch = []
        test_accs_per_epoch = []
        # Tryb treningowy
        model.train()
        for x_train, y_train, weights in loaders['train']:
            x_train, y_train, weights = x_train.to(device), y_train.to(device), weights.to(device)
            y_pred = model(x_train)
            loss_train = criterion(y_pred, y_train)
            loss_train = (loss_train * weights).mean()
            loss_train.backward()
            optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Obliczanie dokładności na zbiorze treningowym
            acc_train = accuracy(y_pred, y_train)
            
            train_losses_per_epoch.append(loss_train.item())
            train_accs_per_epoch.append(acc_train)
            
        train_loss = mean(train_losses_per_epoch)
        train_acc = mean(train_accs_per_epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Tryb ewaluacji dla zbioru testowego
        model.eval()
        with torch.no_grad():
            for x_test, y_test in loaders['test']:
                x_test, y_test = x_test.to(device), y_test.to(device)
                y_pred = model(x_test)
                loss_test = criterion(y_pred, y_test)
                acc_test = accuracy(y_pred, y_test)
                
                test_losses_per_epoch.append(loss_test.item())
                test_accs_per_epoch.append(acc_test)
        
            test_loss = mean(test_losses_per_epoch)
            test_acc = mean(test_accs_per_epoch)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        
        
        
        print(f"Epoka {epoch+1}/{config.n_epochs}: Strata treningowa: {train_loss:.4f}, Dokładność treningowa: {train_acc:.4f}, Strata testowa: {test_loss:.4f}, Dokładność testowa: {test_acc:.4f}")
    
    
    return train_losses, test_losses, train_accs, test_accs


def main(config):
    # Ustawienie ziarna losowości
    set_seed(config.seed)

    # Przygotowanie loaderów
    loaders = prepare_loaders(config.dataset_name, config.dataset_params, config.loader_params)
    
    # Przygotowanie modelu
    model = prepare_model(config.model_name, config.model_params, model_path=config.load_checkpoint_path).to(device)
    
    # Przygotowanie optymalizatora
    optimizer, lr_scheduler = prepare_optim_and_scheduler(model, config.optim_name, config.optim_params, config.scheduler_name, config.scheduler_params)
    
    # Definicja funkcji straty (cross entropy)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Trenowanie modelu
    train_losses, test_losses, train_acc, test_acc = train_model(
        loaders, model, optimizer, criterion,
        config
    )
    
    if config.save_checkpoint_path in not None:
        save_checkpoint(model, optimizer, epoch+1, filename=f"{config.save_checkpoint_path}_epoch{epoch+1}.pth")
    
    matplotlib_scatters_training(config.n_epochs, train_losses, test_losses, train_acc, test_acc)
        
    # Zapisywanie konfiguracji, metryk treningowych oraz metryk wartości osobliwych
    to_save = {
        'config': config,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
    }
    torch.save(to_save, os.path.join("results", "info.pth"))


if __name__ == "__main__":
    # Przykładowa konfiguracja
    class Config:
        seed = 83
        dataset_name = 'cifar10'    # loaders prepare params     
        dataset_params = {'dataset_path': None, 'error_weight': 3, 'failure_indices_path': None}
        loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 12}
        model_name = 'resnet18'     # model prepare params
        model_params = {'num_classes': 10}
        model_path=None
        init=None
        optim_name = 'sgd'      # optim prepare params
        optim_params = {'lr': 1e-1, 'weight_decay': 0.0}
        scheduler_name = None
        scheduler_params = None
        n_epochs = 100              # Liczba epok treningowych
        postfix_title = 'ood_testset_new'
        save_checkpoint_path = None
        load_checkpoint_path = None
        
    
    config = Config()
    main(config)
