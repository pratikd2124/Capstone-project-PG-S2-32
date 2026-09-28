"""The network architecture and the Gamma-KL divergence loss used to train it.

Identical to ``net_dataset_init.py``'s ``Net`` class and ``misc_functions.K`` / ``misc_functions.KL``.
Kept free of any training-loop or dataset code so it can be imported without pulling in torch's
DataLoader machinery (e.g. from inference-adjacent code, or tests).
"""
from __future__ import annotations

from collections import OrderedDict
from itertools import chain
from typing import Dict

import torch
import torch.nn as nn


class Net(nn.Module):
    """Identical to the original ``net_dataset_init.py`` -> ``Net`` class.

    A stack of ``n_layers`` linear+ReLU blocks of width ``width``, mapping ``n_features``
    inputs to 2 outputs (normalised shape, rate of the predicted Gamma unreliability).
    """

    def __init__(self, n_features: int, n_layers: int, width: int):
        super().__init__()
        inner_layers_list = [
            [(f"linear{l}", nn.Linear(width, width)), (f"relu{l}", nn.ReLU())]
            for l in range(2, n_layers + 1)
        ]
        inner_layers_list = [i for i in chain.from_iterable(inner_layers_list)]
        od = OrderedDict(
            [("linear1", nn.Linear(n_features, width)), ("relu1", nn.ReLU())]
            + inner_layers_list
            + [("linear_out", nn.Linear(width, 2))]
        )
        self.linear_relu_stack = nn.Sequential(od)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)


def K(shape_1: torch.Tensor, rate_1: torch.Tensor, shape_2: torch.Tensor, rate_2: torch.Tensor) -> torch.Tensor:
    """Identical to ``misc_functions.K``: one component of the KL divergence between two Gammas."""
    shape_1 = torch.abs(shape_1)
    rate_1 = torch.abs(rate_1)
    a = 1 / rate_1
    b = shape_1
    c = 1 / rate_2
    d = shape_2
    return -c * d / a - b * torch.log(a) - torch.lgamma(b) + (b - 1) * (torch.digamma(d) + torch.log(c))


def KL(
    shape_rate_min_max: Dict[str, float],
    output: torch.Tensor,
    target: torch.Tensor,
    reduce_mean: bool = True,
) -> torch.Tensor:
    """Identical to ``misc_functions.KL``: the training loss (KL divergence between predicted
    and true Gamma distributions, on the RAW/unnormalised shape-rate scale).

    Predictions with a negative (post-rescaling) shape or rate fall back to a squared-error
    penalty instead of the (undefined) KL divergence.
    """
    srmm = shape_rate_min_max
    shape_p = (srmm["shape_max"] - srmm["shape_min"]) * output[:, 0] + srmm["shape_min"]
    rate_p = (srmm["rate_max"] - srmm["rate_min"]) * output[:, 1] + srmm["rate_min"]
    shape_t = (srmm["shape_max"] - srmm["shape_min"]) * target[:, 0] + srmm["shape_min"]
    rate_t = (srmm["rate_max"] - srmm["rate_min"]) * target[:, 1] + srmm["rate_min"]

    K_tt = K(shape_t, rate_t, shape_t, rate_t)
    K_pt = K(shape_p, rate_p, shape_t, rate_t)
    K_diff = K_tt - K_pt

    neg_shape = shape_p < 0
    neg_rate = rate_p < 0
    K_diff = K_diff.clone()
    K_diff[neg_shape] = 5 * (shape_p[neg_shape] - shape_t[neg_shape]) ** 2
    K_diff[neg_rate] = 5 * (rate_p[neg_rate] - rate_t[neg_rate]) ** 2

    return torch.mean(K_diff) if reduce_mean else K_diff
