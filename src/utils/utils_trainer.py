import os
from datetime import datetime


def update_metrics(epochs_metrics, running_metrics): # uprość
    for metric_name in running_metrics:
        if metric_name == 'batch_sizes':
            continue
        if metric_name.endswith('@'):
            names = metric_name[:-1].split('_')
            group_name = "_".join(names[:-1])
            metric_values = sum(running_metrics[metric_name]) / (1 if metric_name.endswith('sizes@') else sum(running_metrics[group_name + "_batch_sizes@"]))
        else:
            metric_values = sum(running_metrics[metric_name]) / sum(running_metrics['batch_sizes'])
        if metric_name not in epochs_metrics:
            epochs_metrics[metric_name] = []
        epochs_metrics[metric_name].append(metric_values)


def adjust_to_log(metrics: dict[str, list[int]], scope: str, window_start: int, round_at: int=5) -> dict[str, int]: # uprość
    logs = {}
    for metric_name in metrics:
        if metric_name in ('batch_sizes', 'epoch') or (len(metrics[metric_name]) < window_start):   # batch_sizes -> batchsizes ?
            continue
        if metric_name.endswith('@'):
            names = metric_name[:-1].split('_')
            group_name = "_".join(names[:-1])
            metric_name_new = names[-1]
            key_new = f'{scope}_{metric_name_new}_subgroup/{group_name}'
            logs[key_new] = (
                sum(metrics[metric_name][-window_start:])
                / (1 if metric_name.endswith('sizes@') 
                   else sum(metrics[group_name + "_batch_sizes@"][-window_start:]))
                if scope == 'running'
                else metrics[metric_name][-1]
            )
        else:
            names = metric_name.split('_')
            phase_new = "".join(names[:-1])
            metric_name_new = names[-1]
            key_new = f'{scope}_{metric_name_new}/{phase_new}'
            logs[key_new] = (
                sum(metrics[metric_name][-window_start:])
                / sum(metrics['batch_sizes'][-window_start:])
                if scope == 'running'
                else metrics[metric_name][-1]
            )
        logs[key_new] = round(logs[key_new], round_at)
    return logs


def create_paths(base_path, exp_name):
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(os.getcwd(), f'{base_path}/{exp_name}/{date}')
    base_save_path = f'{base_path}/checkpoints'
    os.makedirs(base_save_path)
    return base_path, base_save_path
