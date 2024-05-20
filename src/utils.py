import copy
import time
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch


@dataclass
class Timer:
    def __init__(self):
        self.start_time: float | None
        self.elapsed_time: float

    def __enter__(self):
        self.start_time = time.perf_counter()

        return self

    def __exit__(self, *exc_info):
        assert self.start_time is not None
        self.elapsed_time = time.perf_counter() - self.start_time
        self.start_time = None


def weighted_avg(
    sds: list[dict[str, Any]],
    weights: List | np.ndarray,
) -> dict[str, torch.Tensor]:
    """Calculate the weighted average of a list of state dictionaries.

    Args:
    ----
        sds (list[dict[str, Any]]): A list of state dictionaries containing the weights and values.
        weights (List | np.ndarray): The weights to be applied to the state dictionaries.

    Returns:
    -------
        dict[str, torch.Tensor]: The weighted average of the state dictionaries.

    """
    # Normalize the weights.
    weights = softmax_scale(np.array(weights))

    # Calculate the weighted average, layer by layer.
    avg_sd = copy.deepcopy(sds[0])
    for layer in avg_sd:
        avg_sd[layer] = 0.0
        for sd, weight in zip(sds, weights):
            avg_sd[layer] += weight * sd[layer]

    return avg_sd


def softmax_scale(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Scales the input array `x` using the softmax normalization technique.

    Args:
    ----
        x (np.ndarray): The input array to be scaled.
        tau (float): The temperature parameter for the softmax function.

    Returns:
    -------
        np.ndarray: The scaled array after applying the softmax normalization.

    """
    return np.exp(x * tau) / sum(np.exp(x * tau))


def minmax_scale(x: np.ndarray) -> np.ndarray:
    """Scales the input array `x` using the min-max normalization technique.

    Parameters
    ----------
        x (np.ndarray): The input array to be scaled.

    Returns
    -------
        np.ndarray: The scaled array after applying the min-max normalization.

    """
    return (x - x.min()) / (x.max() - x.min())


def compute_model_size(model: torch.nn.Module) -> int:
    """Computes the size of the given model's parameters and buffers in bytes.

    Parameters
    ----------
        model (torch.nn.Module): The input model to compute the size.

    Returns
    -------
        int: The total size of the model in bytes.

    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return int(param_size + buffer_size)
