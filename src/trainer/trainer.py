import logging
from typing import Dict

import torch

from src.visualization.wandb_logger import WandbLogger
from src.utils.utils_general import save_checkpoint, save_training_artefacts
from src.utils.utils_metrics import accuracy
from src.utils.utils_trainer import create_paths, update_metrics, adjust_to_log
from src.utils.utils_visualize import matplotlib_scatters_training, log_to_console, show_and_save_grid
from src.utils.utils_data import save_failure_indices


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
        self.base_save_path = None
        self.global_step = None

        self.extra_modules = extra_modules


    def at_exp_start(self, config):
            """
            Initialization of experiment.
            Creates fullname, dirs and logger.
            """
            self.base_path, self.base_save_path = create_paths(config.trainer_params['base_path'], config.trainer_params['exp_name'])
            config.logger_params['log_dir'] = f'{self.base_path}/{config.logger_params["logger_name"]}'
            self.logger = WandbLogger(config.logger_params, exp_name=config.trainer_params['exp_name'])
            
            self.logger.log_model(self.model, self.criterions['train'], log=None)

            for phase, loader in self.loaders.items():
                show_and_save_grid(loader.dataset, save_path=f"{self.base_path}/{phase}_dataset.png")


    def train_model(self, config):
        logging.info('Training started.')

        self.at_exp_start(config)
        
        # 1) Przygotowanie metryk
        epochs_metrics = {}
        
        for epoch in range(config.trainer_params['n_epochs']):
            epochs_metrics['epoch'] = epoch # czy musze to zapisywać?

            # 1) Zapisz stanu modelu i optymalizatora
            if (epoch > 0) and (epoch % config.trainer_params['save_checkpoint_modulo'] == 0):
                save_checkpoint(
                    self.model,
                    self.optim,
                    epochs_metrics['epoch'],
                    save_path=f"{self.base_save_path}/checkpoint_epoch_{epochs_metrics['epoch']}.pth"
                )
                if config.data_params['dataset_params']['failure_indices_path'] is None:
                    save_failure_indices(
                        self.model,
                        self.loaders['train'].dataset,
                        save_path=f"{self.base_save_path}/failure_indices_epoch_{epochs_metrics['epoch']}",
                        device=config.trainer_params['device']
                    )
            
            # 2) Faza treningowa
            self.model.train()
            self.run_phase(epochs_metrics, phase='train', config=config)
            
            # 3) Faza ewaluacji (wszystkie oprócz 'train')
            with torch.no_grad():
                self.model.eval()
                for phase in self.loaders:
                    if phase == 'train': continue
                    self.run_phase(epochs_metrics, phase=phase, config=config)

            # ════════════════════════ logging (epoch) ════════════════════════ #
            
            epoch_logs = adjust_to_log(epochs_metrics, scope='epoch', window_start=0)
            self.log(
                epoch_logs,
                phase='test',
                scope='epoch',
                step=epoch
            )
            
            # 4) Logowanie metryk do konsoli
            log_to_console(epochs_metrics)
        
        self.at_exp_end(config, epochs_metrics)
    

    def run_phase(self, epochs_metrics, phase, config):
        logging.info(f'Epoch: {epochs_metrics['epoch']}, Phase: {phase}.')
        
        running_metrics = {
            f"{phase}_{metric_name}": []
            for metric_name in ('losses', 'accs')
        }
        running_metrics['batch_sizes'] = []

        batches_per_epoch = len(self.loaders[phase])
        self.global_step = epochs_metrics['epoch'] * batches_per_epoch  # czy musi tu być self.?
        config.trainer_params['running_window_start'] = batches_per_epoch // 10
        
        for i, data in enumerate(self.loaders[phase]):
            y_pred, y_true, aux_data = self.infer_from_data(data, device=config.trainer_params['device'])
            running_metrics = self.gather_batch_metrics(phase, running_metrics, y_pred, y_true, aux_data)   # cos nie działa podczas testowania

            # ════════════════════════ logging (running) ════════════════════════ #

            if (i + 1) % config.trainer_params['running_window_start'] == 0: # lepsza nazwa na log_multi? słowniek multi = {'log'...}?
                # przygotuj metryki do logowania (running)
                running_logs = adjust_to_log(running_metrics, scope='running', window_start=config.trainer_params['running_window_start'])
                self.log(
                    running_logs,
                    phase,
                    scope='running',
                    step=self.global_step
                )

            self.global_step += 1
        
        update_metrics(epochs_metrics, running_metrics)
        

    def log(self, scope_logs: Dict, phase: str, scope: str, step: int):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict):
            phase:
            scope:
            progress_bar:
        '''
        scope_logs[f'steps/{phase}_{scope}'] = step
        self.logger.log_scalars(scope_logs, step)
        # progress_bar.set_postfix(evaluators_log)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)


    def at_exp_end(self, config, epochs_metrics):
        save_training_artefacts(
            config,
            epochs_metrics,
            save_path=f"{self.base_save_path}/training_artefacts.pth"
        )

        save_checkpoint(
            self.model,
            self.optim,
            epochs_metrics['epoch'],
            save_path=f"{self.base_save_path}/epoch_{epochs_metrics['epoch']}.pth"
        )
            
        matplotlib_scatters_training(epochs_metrics, save_path=f"{self.base_path}/metrics.pdf")

        if config.data_params['dataset_params']['failure_indices_path'] is None:
            save_failure_indices(
                self.model,
                self.loaders['train'].dataset,
                save_path=f"{self.base_save_path}/failure_indices",
                device=config.trainer_params['device']
            )


    def infer_from_data(self, data, device):
        x_true, y_true, aux_data = data
        # x_true, y_true = x_true.to(device)[:20], y_true.to(device)[:20]
        x_true, y_true = x_true.to(device), y_true.to(device)
        for i in range(len(aux_data)):
            # aux_data[i] = aux_data[i].to(device)[:20]
            aux_data[i] = aux_data[i].to(device)
        y_pred = self.model(x_true)
        return y_pred, y_true, aux_data
    
    
    def gather_batch_metrics(self, phase, running_metrics, y_pred, y_true, aux_data):
        loss_list = self.criterions['train' if phase == 'train' else 'eval'](y_pred, y_true)

        if len(aux_data) == 2:  # co gdy len > 2?
            is_corrupted, sample_weights = aux_data
            loss = (loss_list * sample_weights).mean()
        else:
            is_corrupted = aux_data[0]
            loss = loss_list.mean()
            
        if phase == 'train':
            loss.backward()
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
        
        acc = accuracy(y_pred, y_true)
        batch_size = y_true.shape[0]

        # ════════════════════════ gathering scalars to logging ════════════════════════ #
        
        if len(aux_data) == 2:
            loss, loss_weighed = loss_list.mean(), loss
            if f'{phase}_lossesweighted' not in running_metrics:
                running_metrics[f'{phase}_lossesweighted'] = []
            running_metrics[f'{phase}_lossesweighted'].append(loss_weighed.item() * batch_size)
        
        running_metrics[f'{phase}_losses'].append(loss.item() * batch_size)
        running_metrics[f'{phase}_accs'].append(acc * batch_size)
        running_metrics['batch_sizes'].append(batch_size)

        for cls in [0, 1]: # trzeba rozszerzyć do niearbitrarnych klas
            for corrupted in [True, False]:
                mask = (y_true == cls) & (is_corrupted == corrupted)
                if mask.sum() == 0: continue  # brak przykładów tej grupy w tym batchu

                group_name = f"{phase}_cls{cls}{'cor' if corrupted else 'clean'}"
                loss_i = loss_list[mask].mean().item()
                acc_i = accuracy(y_pred[mask], y_true[mask])

                if group_name + "_losses@" not in running_metrics:
                    running_metrics[group_name + "_losses@"] = []
                    running_metrics[group_name + "_accs@"] = []
                    running_metrics[group_name + "_batch_sizes@"] = []

                subgroup_size = mask.sum().item()
                running_metrics[group_name + "_losses@"].append(loss_i * subgroup_size)
                running_metrics[group_name + "_accs@"].append(acc_i * subgroup_size)
                running_metrics[group_name + "_batch_sizes@"].append(subgroup_size)

        return running_metrics