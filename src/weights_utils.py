import copy

import torch


def vectorize_checkpoint(checkpoint, layer_lst, use_bias=False):
    """Flatten the selected model parameters in layer order."""
    parameters = []
    for layer, _ in layer_lst:
        parameters.append(checkpoint[f"module_list.{layer}.weight"].reshape(-1))
        if use_bias:
            parameters.append(checkpoint[f"module_list.{layer}.bias"].reshape(-1))
    if not parameters:
        return torch.empty(0)
    return torch.cat(parameters)


def vector_to_checkpoint(checkpoint, vector, layer_lst, use_bias=False):
    """Restore selected parameters from a flat vector into a checkpoint copy."""
    if vector.ndim != 1:
        raise ValueError(f"expected a 1D vector, got shape {tuple(vector.shape)}")

    restored = copy.deepcopy(checkpoint)
    offset = 0
    for layer, _ in layer_lst:
        parameter_names = ("weight", "bias") if use_bias else ("weight",)
        for parameter_name in parameter_names:
            key = f"module_list.{layer}.{parameter_name}"
            original = restored[key]
            end = offset + original.numel()
            if end > vector.numel():
                raise ValueError("vector is shorter than the selected checkpoint parameters")
            restored[key] = vector[offset:end].reshape_as(original).clone()
            offset = end

    if offset != vector.numel():
        raise ValueError(
            f"vector has {vector.numel() - offset} unused values after reconstruction"
        )
    return restored


def add_noise_to_checkpoint(checkpoint, layer_lst, use_bias=False, sigma=0.01):
    """Add Gaussian noise in place to the selected checkpoint parameters."""
    with torch.no_grad():
        for layer, _ in layer_lst:
            weight = checkpoint[f"module_list.{layer}.weight"]
            weight.add_(sigma * torch.randn_like(weight))
            if use_bias:
                bias = checkpoint[f"module_list.{layer}.bias"]
                bias.add_(sigma * torch.randn_like(bias))
    return checkpoint
