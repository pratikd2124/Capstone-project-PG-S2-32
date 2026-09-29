"""Comparing a freshly trained network with RYA's trained EXP002.

Both models score the same test rows, and we report how far each is from the true answers
(mean absolute error of shape). After a quick 15-epoch demo the fresh model is expected to do
much worse than EXP002, which trained for 1,000 epochs. That gap is the point of the demo,
not a bug.

Remember that RYA never saved EXP002's train/test split, so many of these "test" rows were
probably in EXP002's own training data. We need to treat EXP002's number as a reference, not a fair held-out score.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch

from .inference import UnreliabilityPredictor
from .model import Net


def _rescale(min_max_df: pd.DataFrame, col: str, values: np.ndarray) -> np.ndarray:
    """Undo the 0-1 scaling for one column, using its min/max from the scaling table."""
    mn = float(min_max_df.loc[min_max_df["column"] == col, "min"].iloc[0])
    mx = float(min_max_df.loc[min_max_df["column"] == col, "max"].iloc[0])
    return (mx - mn) * values + mn


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error: the average size of the miss."""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def compare_against_exp002(
    demo_model: Net,
    predictor: UnreliabilityPredictor,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    min_max_train_df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, object]:
    """Score the test rows with both models.

    Returns each model's shape and rate predictions, the true values, and the shape MAE of each
    (keys: demo_shape_pred, demo_rate_pred, exp002_shape_pred, exp002_rate_pred, true_shape,
    true_rate, mae_demo_shape, mae_exp002_shape).
    """
    # The fresh model works directly on the scaled test rows.
    demo_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(test_df[feature_columns].values.astype(np.float32)).to(device)
        demo_pred_norm = demo_model(X_test_tensor).cpu().numpy()

    demo_shape_pred = _rescale(min_max_train_df, "shape", demo_pred_norm[:, 0])
    demo_rate_pred = _rescale(min_max_train_df, "rate", demo_pred_norm[:, 1])
    true_shape = _rescale(min_max_train_df, "shape", test_df["shape"].values)
    true_rate = _rescale(min_max_train_df, "rate", test_df["rate"].values)

    # EXP002 has its own scaling (from EXP002.json), so unscale the rows back to real values
    # first and let the predictor rescale them its own way.
    raw_features_df = test_df[feature_columns].copy()
    for c in feature_columns:
        raw_features_df[c] = _rescale(min_max_train_df, c, raw_features_df[c].values)

    exp002_shape_pred, exp002_rate_pred = [], []
    for _, row in raw_features_df.iterrows():
        shape_hat, rate_hat = predictor.predict_raw_features(row)
        exp002_shape_pred.append(shape_hat)
        exp002_rate_pred.append(rate_hat)
    exp002_shape_pred = np.array(exp002_shape_pred)
    exp002_rate_pred = np.array(exp002_rate_pred)

    return {
        "demo_shape_pred": demo_shape_pred,
        "demo_rate_pred": demo_rate_pred,
        "exp002_shape_pred": exp002_shape_pred,
        "exp002_rate_pred": exp002_rate_pred,
        "true_shape": true_shape,
        "true_rate": true_rate,
        "mae_demo_shape": mae(demo_shape_pred, true_shape),
        "mae_exp002_shape": mae(exp002_shape_pred, true_shape),
    }


def plot_comparison(result: Dict[str, object], num_epochs: int, ax=None):
    """Predicted vs true shape for both models, side by side (closer to the line = better)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    true_shape = result["true_shape"]
    for a, (pred, label, color) in zip(
        ax,
        [
            (result["demo_shape_pred"], f"Trained in this run ({num_epochs} epochs)", "tab:orange"),
            (result["exp002_shape_pred"], "Pre-trained EXP002 (1000 epochs)", "tab:blue"),
        ],
    ):
        a.scatter(true_shape, pred, alpha=0.3, s=10, color=color)
        lims = [min(true_shape.min(), pred.min()), max(true_shape.max(), pred.max())]
        a.plot(lims, lims, "k--", linewidth=1)
        a.set_xlabel("true shape")
        a.set_ylabel("predicted shape")
        a.set_title(label)
    return ax
