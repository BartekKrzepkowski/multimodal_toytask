import logging
import os

import torch

from src.trainer.trainer import Trainer
from src.utils.utils_data import prepare_loaders
from src.utils.utils_model import prepare_model
from src.utils.utils_optim import prepare_optim_and_scheduler


from src.utils.utils_general import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"     


def main(config):
     # ════════════════════════ prepare seed ════════════════════════ #


    set_seed(config.trainer_params['seed'])
    logging.info('Random seed prepared.')


    # ════════════════════════ prepare loaders ════════════════════════ #


    loaders = prepare_loaders(config.data_params)
    logging.info('Loaders prepared.')
    

    # ════════════════════════ prepare model ════════════════════════ #


    model = prepare_model(config.model_params).to(device)
    logging.info('Model prepared.')
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #


    optim, lr_scheduler = prepare_optim_and_scheduler(model, config.optim_scheduler_params)
    logging.info('Optimizer and scheduler prepared.')
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #


    criterions = {}
    criterions['train'] = torch.nn.CrossEntropyLoss(reduction='none')
    #     torch.nn.CrossEntropyLoss()
    #     if config.data_params['dataset_params']['failure_indices_path'] is None
    #     else torch.nn.CrossEntropyLoss(reduction='none')
    # )
    criterions['eval'] = torch.nn.CrossEntropyLoss(reduction='none')
    logging.info('Criterion prepared.')


    # ════════════════════════ prepare extra modules ════════════════════════ #
    

    extra_modules = {}


    # ════════════════════════ prepare trainer ════════════════════════ #
    

    params_trainer = {
        'model': model,
        'criterions': criterions,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'extra_modules': extra_modules,
    }
    trainer = Trainer(**params_trainer)
    logging.info('Trainer prepared.')


    # ════════════════════════ train model ════════════════════════ #


    trainer.train_model(config)
    logging.info('Training finished.')


if __name__ == "__main__":
    logging.basicConfig(
            format=(
                '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
            ),
            level=logging.INFO,
            handlers=[logging.StreamHandler()],
            force=True,
        )
    # Przykładowa konfiguracja
    class Config: # przejrzyj ten config, pogrupuj i zadbaj o tytuł eksperymentu
        trainer_params = {
            'device': device,
            'seed': 83,
            'n_epochs': 200,
            'exp_name': 'cats_and_dogs_blurred_blur_prob_0.4', # nazwa eksperymentu, która będzie użyta do tworzenia folderów i logowania
            'base_path': os.environ['REPORTS_DIR'],
            'load_checkpoint_path': None,    # zapisuje checkpoint optymalizatora i modelu, nie tylko samego modelu
            'save_checkpoint_modulo': 10  # co ile epok zapisywać checkpoint
        }
        data_params = {
            'dataset_name' : 'cats_and_dogs_blurred',
            'dataset_params': {
                'dataset_path': None,
                'failure_indices_path': None,#'/net/pr2/projects/plgrid/plggdnnp/bartek/reports/cats_and_dogs_noise/2025-05-12_17-54-09/checkpoints/failure_indices',
                'use_transform': False, # if True, then use transforms for the base dataset (per side [CIFAR10 at this point])
                'blur_prob': 0.4, # probability of applying blur augmentation
                'error_weight': 10,
                'is_blur': True, # if True, then use blur augmentation else use noise augmentation
                'failure_percentage': 1.0   # percentage of failure indices in the training set
            },
            'loader_params': {'batch_size': 125, 'pin_memory': True, 'num_workers': 12}
        }
        model_params = {
            'model_name': 'resnet18',
            'model_params': {'num_classes': 2}, # spróbuj z większą liczbą klas
            'model_path': None, # sprawdz czy to nie problem że zapisuje checkpoint modelu i optimalizatora
            'init': None
        }
        optim_scheduler_params = {
            'optim_name': 'sgd',
            'optim_params': {'lr': 1e-1, 'weight_decay': 0.0},
            'scheduler_name': None,
            'scheduler_params': None
        }
        logger_params = {
            'logger_name': 'wandb',
            'entity': os.environ['WANDB_ENTITY'],
            'project_name': os.environ['WANDB_PROJECT'],
            'mode': 'online',   # używając tego określ również czy logować info na dysk
            # 'hyperparameters': h_params_overall,
        }
         
    config = Config()
    main(config)
