import torch


FORBIDDEN_LAYER_TYPES = [torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]


def get_every_but_forbidden_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_every_but_forbidden_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def configure_optimizer(optim_wrapper, model, optim_kwargs):
    weight_decay = optim_kwargs['weight_decay']
    del optim_kwargs['weight_decay']

    decay_parameters = get_every_but_forbidden_parameter_names(model, FORBIDDEN_LAYER_TYPES)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for pn, p in model.named_parameters() if pn in decay_parameters and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for pn, p in model.named_parameters() if pn not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim_wrapper(optimizer_grouped_parameters, **optim_kwargs)
    return optimizer


# Funkcja przygotowująca optymalizator
def prepare_optim_and_scheduler(model, optim_name, optim_params, scheduler_name=None, scheduler_params=None):
    optim_wrapper = torch.optim.SGD  #OPTIMIZER_NAME_MAP[optim_name]
    optim = configure_optimizer(optim_wrapper, model, optim_params)
    lr_scheduler = SCHEDULER_NAME_MAP[scheduler_name](optim, **scheduler_params) if scheduler_name is not None else None
    return optim, lr_scheduler