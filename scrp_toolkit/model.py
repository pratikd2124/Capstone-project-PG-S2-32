"""The network design and the loss it was trained with.

``Net`` is RYA's network, copied from net_dataset_init_B.py: a straight stack of fully
connected layers with ReLU in between, ending in two outputs (scaled shape and scaled rate).
EXP002 uses 40 inputs, 20 layers and 256 units per layer, about 1.26 million numbers in
total. Walkthrough Step 7 loads the trained weights from the ONNX file into this class and
gets identical outputs, which proves it's the same network.

``K`` and ``KL`` are RYA's training loss from misc_functions.py: how far the predicted Gamma
distribution is from the true one. They're only needed for (re)training experiments. Using
RYA's model as-is doesn't need them.
"""
from __future__ import annotations

from collections import OrderedDict
from itertools import chain
from typing import Dict

import torch
import torch.nn as nn


class Net(nn.Module):
    """RYA's network: ``n_layers`` layers of ``width`` units, ``n_features`` in, 2 out.

    The layer names (linear1, relu1, ..., linear_out) match RYA's exactly, so saved weights
    from their code load straight in.
    """

    def __init__(self, n_features: int, n_layers: int, width: int):
        super().__init__()
        # Layers 2..n: width -> width, each followed by a ReLU.
        inner_layers_list = [
            [(f"linear{l}", nn.Linear(width, width)), (f"relu{l}", nn.ReLU())]
            for l in range(2, n_layers + 1)
        ]
        inner_layers_list = [i for i in chain.from_iterable(inner_layers_list)]
        # First layer takes the inputs; the last layer gives the two outputs.
        od = OrderedDict(
            [("linear1", nn.Linear(n_features, width)), ("relu1", nn.ReLU())]
            + inner_layers_list
            + [("linear_out", nn.Linear(width, 2))]
        )
        self.linear_relu_stack = nn.Sequential(od)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)


def K(shape_1: torch.Tensor, rate_1: torch.Tensor, shape_2: torch.Tensor, rate_2: torch.Tensor) -> torch.Tensor:
    """One piece of the distance between two Gamma distributions (RYA's ``K``).

    Negative inputs are made positive here so the logs don't fail. Those cases are
    replaced by a separate penalty in ``KL`` anyway.
    """
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
    """The training loss: how far each predicted Gamma is from the true one (RYA's ``KL``).

    The network works on scaled values (0-1), so both prediction and target are first turned
    back into real shape/rate using the training min/max. If a prediction comes out negative,
    which a Gamma can't have, that row gets a squared-error penalty instead, pushing the
    network back towards sensible values.

    reduce_mean=False returns one loss per row instead of the average.
    """
    srmm = shape_rate_min_max
    shape_p = (srmm["shape_max"] - srmm["shape_min"]) * output[:, 0] + srmm["shape_min"]
    rate_p = (srmm["rate_max"] - srmm["rate_min"]) * output[:, 1] + srmm["rate_min"]
    shape_t = (srmm["shape_max"] - srmm["shape_min"]) * target[:, 0] + srmm["shape_min"]
    rate_t = (srmm["rate_max"] - srmm["rate_min"]) * target[:, 1] + srmm["rate_min"]

    K_tt = K(shape_t, rate_t, shape_t, rate_t)
    K_pt = K(shape_p, rate_p, shape_t, rate_t)
    K_diff = K_tt - K_pt

    # Replace impossible (negative) predictions with the penalty.
    neg_shape = shape_p < 0
    neg_rate = rate_p < 0
    K_diff = K_diff.clone()
    K_diff[neg_shape] = 5 * (shape_p[neg_shape] - shape_t[neg_shape]) ** 2
    K_diff[neg_rate] = 5 * (rate_p[neg_rate] - rate_t[neg_rate]) ** 2

    return torch.mean(K_diff) if reduce_mean else K_diff
