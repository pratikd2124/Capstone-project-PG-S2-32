"""From-scratch training loop for the ``Net`` architecture, using the Gamma-KL loss.

Faithful port of the notebook's Section 3 training loop (``train_test_loop.py`` /
``perform_exp.py``). The shipped EXP002 model was trained with 20 layers, width 256, for
1000 epochs; QUICK_DEMO trains for far fewer epochs purely so the pipeline can be smoke-tested
in minutes instead of hours -- see Section 4 / ``compare.py`` for why the demo model is
expected to under-perform EXP002.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import DHatTensorDataset, build_preprocessed_dataset, train_test_split_like_original
from .model import KL, Net


@dataclass
class TrainConfig:
    n_layers: int = 20
    width: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    batch_size: int = 64
    train_frac: float = 0.8
    random_seed: int = 42
    device: str | None = None  # None -> auto-detect cuda/cpu

    def resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shape_rate_min_max_from(min_max_train_df: pd.DataFrame) -> dict:
    def _get(col, stat):
        return float(min_max_train_df.loc[min_max_train_df["column"] == col, stat].iloc[0])

    return {
        "shape_min": _get("shape", "min"),
        "shape_max": _get("shape", "max"),
        "rate_min": _get("rate", "min"),
        "rate_max": _get("rate", "max"),
    }


def run_epoch(loader: DataLoader, model: Net, loss_fn, device: torch.device, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_seen = 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * len(X)
            n_seen += len(X)
    return total_loss / n_seen


def train_from_instance_data(instance_dir: str | None, config: TrainConfig | None = None):
    """End-to-end: build the dataset, train a fresh ``Net``, return everything needed downstream.

    Returns a dict with: model, config, device, feature_columns, min_max_train_df,
    train_df, test_df, train_loss_history, test_loss_history.
    """
    if not instance_dir:
        raise FileNotFoundError(
            "No instance_data/ folder was found under the project root -- training needs the raw "
            "orthog_B_*.csv feature files."
        )
    config = config or TrainConfig()
    device = config.resolve_device()

    full_df, feature_columns, min_max_train_df = build_preprocessed_dataset(instance_dir)
    train_df, test_df = train_test_split_like_original(
        full_df, train_frac=config.train_frac, random_seed=config.random_seed
    )

    train_ds = DHatTensorDataset(train_df, feature_columns)
    test_ds = DHatTensorDataset(test_df, feature_columns)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=True)

    shape_rate_min_max = shape_rate_min_max_from(min_max_train_df)
    model = Net(len(feature_columns), config.n_layers, config.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = partial(KL, shape_rate_min_max)

    train_loss_history: List[float] = []
    test_loss_history: List[float] = []
    for epoch in range(config.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, device, optimizer)
        test_loss = run_epoch(test_loader, model, loss_fn, device, optimizer=None)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        log_every = max(1, config.num_epochs // 15)
        if epoch % log_every == 0 or epoch == config.num_epochs - 1:
            print(f"Epoch {epoch + 1:4d}/{config.num_epochs}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")

    return {
        "model": model,
        "config": config,
        "device": device,
        "feature_columns": feature_columns,
        "min_max_train_df": min_max_train_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
    }


def plot_loss_curve(train_loss_history: List[float], test_loss_history: List[float], ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss_history, label="train")
    ax.plot(test_loss_history, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL(Gamma) loss")
    ax.set_title("From-scratch training: loss curve")
    ax.legend()
    return ax
