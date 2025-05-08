import os
from datetime import datetime

def update_metrics(epochs_metrics, running_metrics):
    for metric_name in running_metrics:
        if metric_name == 'batch_sizes':
            continue
        metric_values = sum(running_metrics[metric_name]) / sum(running_metrics['batch_sizes'])
        epochs_metrics[metric_name].append(metric_values)
    # print(f"Zaktualizowano metryki: {epochs_metrics}")

def adjust_to_log(metrics, scope, window_start, round_at=4):
    if scope == 'epoch':
        print(metrics)
    logs = {}
    for metric_name in metrics:
        if metric_name in ('batch_sizes', 'epoch') or len(metrics[metric_name]) < window_start:
            continue
        names = metric_name.split('_')
        new_metric_name = f'{scope}_{names[1]}/{names[0]}'
        logs[new_metric_name] = (
            sum(metrics[metric_name][-window_start:])
            / sum(metrics['batch_sizes'][-window_start:])
            if scope == 'running'
            else metrics[metric_name][-1]
        )
        logs[new_metric_name] = round(logs[new_metric_name], round_at)
    return logs

def create_paths(base_path, exp_name):
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(os.getcwd(), f'{base_path}/{exp_name}/{date}')
    save_path_base = f'{base_path}/checkpoints'
    os.makedirs(save_path_base)
    save_path = lambda step: f'{save_path_base}/model_step_{step}.pth'
    return base_path, save_path

