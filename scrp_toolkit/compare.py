"""Compare a freshly-trained network against the shipped, pre-trained EXP002 ONNX model.

Faithful port of the notebook's Section 4. Both models are evaluated on the same held-out
test set; with a short (QUICK_DEMO-style) training run the from-scratch model is *expected*
to score worse than EXP002 (trained for 1000 epochs) -- that gap is the demonstration, not a bug.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch

from .inference import UnreliabilityPredictor
from .model import Net


def _rescale(min_max_df: pd.DataFrame, col: str, values: np.ndarray) -> np.ndarray:
    mn = float(min_max_df.loc[min_max_df["column"] == col, "min"].iloc[0])
    mx = float(min_max_df.loc[min_max_df["column"] == col, "max"].iloc[0])
    return (mx - mn) * values + mn


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def compare_against_exp002(
    demo_model: Net,
    predictor: UnreliabilityPredictor,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    min_max_train_df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, object]:
    """Score both models on ``test_df`` and return predictions + MAE for each.

    Returns a dict with: demo_shape_pred, demo_rate_pred, exp002_shape_pred, exp002_rate_pred,
    true_shape, true_rate, mae_demo_shape, mae_exp002_shape.
    """
    demo_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(test_df[feature_columns].values.astype(np.float32)).to(device)
        demo_pred_norm = demo_model(X_test_tensor).cpu().numpy()

    demo_shape_pred = _rescale(min_max_train_df, "shape", demo_pred_norm[:, 0])
    demo_rate_pred = _rescale(min_max_train_df, "rate", demo_pred_norm[:, 1])
    true_shape = _rescale(min_max_train_df, "shape", test_df["shape"].values)
    true_rate = _rescale(min_max_train_df, "rate", test_df["rate"].values)

    # test_df's features are normalised to the demo-training scale; convert back to raw values,
    # then re-normalise using EXP002's OWN min-max scale, because the two models were trained on
    # different datasets/scales.
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
