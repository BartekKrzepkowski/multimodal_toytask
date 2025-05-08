import logging
from typing import Dict

import torch

from src.visualization.wandb_logger import WandbLogger
from src.utils.utils_general import save_checkpoint, save_training_artefacts
from src.utils.utils_metrics import accuracy
from src.utils.utils_trainer import create_paths, update_metrics, adjust_to_log
from src.utils.utils_visualize import matplotlib_scatters_training, log_to_console

class Trainer:
    def __init__(self, model, criterions, loaders, optim, lr_scheduler, extra_modules, device):
        self.model = model
        self.criterions = criterions
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.logger = None
        self.base_path = None
        self.save_path = None
        self.epoch = -1
        self.global_step = None

        self.extra_modules = extra_modules


    def at_exp_start(self, config):
            """
            Initialization of experiment.
            Creates fullname, dirs and logger.
            """
            self.base_path, self.save_path = create_paths(config.base_path, config.exp_name)
            config.logger_config['log_dir'] = f'{self.base_path}/{config.logger_config["logger_name"]}'
            self.logger = WandbLogger(config)
            
            self.logger.log_model(self.model, self.criterions['train'], log=None)


    def train_model(self, config):
        logging.info('Training started.')

        self.at_exp_start(config)
        
        # 1) Przygotowanie metryk
        epochs_metrics = {
            f"{phase}_{metric_name}": []
            for phase in self.loaders.keys()
            for metric_name in ('losses', 'accs')
        }
        
        for epoch in range(config.n_epochs):
            epochs_metrics['epoch'] = epoch
            
            # 2) Wybór funkcji trenowania
            train_fn = self.run_phase
            # train_fn = (
            #     self.run_phase
            #     if config.dataset_params['failure_indices_path'] is None
            #     else self.run_weighted_phase sdas #TODO
            # )
            
            # 3) Faza treningowa
            self.model.train()
            train_fn(epochs_metrics, phase='train', config=config)
            
            # 4) Faza ewaluacji (wszystkie oprócz 'train')
            with torch.no_grad():
                self.model.eval()
                for phase in self.loaders:
                    if phase == 'train':
                        continue
                    self.run_phase(epochs_metrics, phase=phase, config=config)
            self.log(
                epoch_logs,
                phase,
                scope='epoch',
                step=epochs_metrics['epoch']
            )
            
            # 5) Logowanie metryk
            log_to_console(epochs_metrics)
        
        save_training_artefacts(config, epochs_metrics, save_path=self.save_path(epoch))
        if config.save_checkpoint_path is not None:
            save_checkpoint(
                self.model,
                self.optim,
                config.epoch+1,
                filename=f"{config.save_checkpoint_path}_epoch{config.epoch+1}.pth"
            )
            
        matplotlib_scatters_training(config, epochs_metrics)
    

    def run_phase(self, epochs_metrics, phase, config):
        logging.info(f'Epoch: {epochs_metrics['epoch']}, Plain phase: {phase}.')
        # logging.info(f'Running plain phase - {phase}.')
        
        running_metrics = {
            f"{phase}_{metric_name}": []
            for metric_name in ('losses', 'accs')
        }
        running_metrics['batch_sizes'] = []

        batches_per_epoch = len(self.loaders[phase])
        self.global_step = epochs_metrics['epoch'] * batches_per_epoch
        config.running_window_start = batches_per_epoch // 10
        
        for i, (x_true, y_true) in enumerate(self.loaders[phase]):
            x_true, y_true = x_true.to(config.device), y_true.to(config.device)
            y_pred = self.model(x_true)
            loss = self.criterions['train' if phase == 'train' else 'eval'](y_pred, y_true)
            
            if phase == 'train':
                loss.backward()
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
            
            acc = accuracy(y_pred, y_true)
            
            batch_size = x_true.shape[0]
            running_metrics[f'{phase}_losses'].append(loss.item() * batch_size)
            running_metrics[f'{phase}_accs'].append(acc * batch_size)
            running_metrics['batch_sizes'].append(batch_size)

            # ════════════════════════ logging (running) ════════════════════════ #

            if (i + 1) % config.running_window_start == 0: # lepsza nazwa na log_multi? słowniek multi = {'log'...}?
                # przygotuj metryki do logowania (running)
                running_logs = adjust_to_log(running_metrics, scope='running', window_start=config.running_window_start)
                self.log(
                    running_logs,
                    phase,
                    scope='running',
                    step=self.global_step
                )

            self.global_step += 1
            
        # ════════════════════════ logging (epoch) ════════════════════════ #
        
        update_metrics(epochs_metrics, running_metrics)
        epoch_logs = adjust_to_log(epochs_metrics, scope='epoch', window_start=0)


    def log(self, scope_logs: Dict, phase: str, scope: str, step: int):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict):
            phase:
            scope:
            progress_bar:
        '''
        # evaluators_log = adjust_evaluators_pre_log(logs['evaluators'], assets['denom'], round_at=4)
        scope_logs[f'steps/{phase}_{scope}'] = step
        self.logger.log_scalars(scope_logs, step)
        # progress_bar.set_postfix(evaluators_log)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)
        