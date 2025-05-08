import logging
import os

import torch

from src.trainer.trainer import Trainer
from src.utils.utils_data import prepare_loaders, get_failure_indices
from src.utils.utils_model import prepare_model
from src.utils.utils_optim import prepare_optim_and_scheduler


from src.utils.utils_general import set_seed, save_checkpoint, save_training_artefacts


device = "cuda" if torch.cuda.is_available() else "cpu"

# def run_weighted_phase(loader, model, optimizer, criterion, epochs_metrics, phase):
#     logging.info(f'Running weighted phase - {phase}.')

#     running_metrics = {
#         f'{phase}_losses': [],
#         f'{phase}_accs': [],
#         'batch_sizes': [],
#     }

#     for x_true, y_true, weights in loader:
#         x_true, y_true, weights = x_true.to(device), y_true.to(device), weights.to(device)
#         y_pred = model(x_true)
#         loss = criterion(y_pred, y_true)
#         loss = (loss * weights).mean()

#         if phase == 'train':
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad(set_to_none=True)
        
#         # Obliczanie dokładności na zbiorze treningowym
#         acc = accuracy(y_pred, y_true)
        
#         batch_size = x_true.shape[0]
#         running_metrics[f'{phase}_losses'].append(loss.item() * batch_size)
#         running_metrics[f'{phase}_accs'].append(acc * batch_size)
#         running_metrics['batch_sizes'].append(batch_size)
        
#     update_metrics(epochs_metrics, running_metrics)
        



def main(config):
     # ════════════════════════ prepare seed ════════════════════════ #


    set_seed(config.seed)
    logging.info('Random seed prepared.')


    # ════════════════════════ prepare loaders ════════════════════════ #


    loaders = prepare_loaders(
        config.dataset_name,
        config.dataset_params,
        config.loader_params
    )
    logging.info('Loaders prepared.')
    

    # ════════════════════════ prepare model ════════════════════════ #


    model = prepare_model(
        config.model_name,
        config.model_params,
        model_path=config.load_checkpoint_path
    ).to(device)
    logging.info('Model prepared.')
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #


    optim, lr_scheduler = prepare_optim_and_scheduler(
        model,
        config.optim_name,
        config.optim_params,
        config.scheduler_name,
        config.scheduler_params
    )
    logging.info('Optimizer and scheduler prepared.')
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #


    criterions = {}
    criterions['train'] = (
        torch.nn.CrossEntropyLoss()
        if config.dataset_params['failure_indices_path'] is None
        else torch.nn.CrossEntropyLoss(reduction='none')
    )
    criterions['eval'] = torch.nn.CrossEntropyLoss()
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
 

    # ════════════════════════ save model ════════════════════════ #


    if config.save_checkpoint_path is not None:
        save_checkpoint(
            model,
            optim,
            config.epoch+1,
            filename=f"{config.save_checkpoint_path}_epoch{config.epoch+1}.pth"
        )
    
    if (
        config.failure_indices_path is not None 
        and config.dataset_params['failure_indices_path'] is None
    ):
        get_failure_indices(
            model,
            loaders['train'].dataset,
            config.failure_indices_path,
            device
        )


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
    class Config:
        seed = 83
        device = device
        exp_name = 'cats_and_dogs_blurred_white_noise_new'
        base_path = os.environ['REPORTS_DIR']
        dataset_name = 'cats_and_dogs_blurred'    # loaders prepare params     
        # dataset_params = {'dataset_path': None, 'dominance_ratio': 0.8, 'error_weight': 3, 'failure_indices_path': 'data/failure_indices_path_2025-04-25_07-31-55', 
        #                   'train_indices_file': None, 'test_indices_file': None, "test_ood_indices_file": None}
        dataset_params = {'dataset_path': None, 'failure_indices_path': None, 'use_transform': False, 'blur_prob': 1.0, 'error_weight': 5, 'is_blur': True}
        loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 12}
        model_name = 'resnet18'     # model prepare params
        model_params = {'num_classes': 2}
        model_path=None
        init=None
        optim_name = 'sgd'      # optim prepare params
        optim_params = {'lr': 1e-1, 'weight_decay': 0.0}
        scheduler_name = None
        scheduler_params = None
        n_epochs = 160              # Liczba epok treningowych
        postfix_title = 'cats_and_dogs_blurred_white_noise_new'
        save_checkpoint_path = None
        load_checkpoint_path = None
        failure_indices_path = 'data/failure_indices_path'
        logger_config = {
            'logger_name': 'wandb',
            'entity': os.environ['WANDB_ENTITY'],
            'project_name': os.environ['WANDB_PROJECT'],
            # 'hyperparameters': h_params_overall,
            'mode': 'online',
        }
        # running_window_start = 
        # log_multi=
        
        
    
    config = Config()
    main(config)
