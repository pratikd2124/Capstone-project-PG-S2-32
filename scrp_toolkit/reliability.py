"""P-matrix (reliability matrix) loading and asymmetry scoring.

Identical logic to the original ``misc_functions.asymm_calc`` / the notebook's Section 1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def asymm_calc(P: np.ndarray) -> float:
    """Mean squared asymmetry of the off-diagonal cells of a 4x4 reliability matrix.

    The more dominant the diagonal (P close to 1 on it) the more consistent the item;
    a larger asymm score means a noisier / less reliable item.
    """
    return sum((P[i, j] - P[j, i]) ** 2 for i, j in _PAIRS) / len(_PAIRS)


def load_p_matrices(pmatrices_path: str | Path) -> Dict[str, np.ndarray]:
    """Load ``unreliability-matrices.json`` and return {question: 4x4 numpy array}."""
    with open(pmatrices_path) as f:
        raw = json.load(f)
    return {q: np.array(P) for q, P in raw.items()}


def score_asymmetry(p_matrices: Dict[str, np.ndarray]) -> pd.Series:
    """Return a Series of asymm scores per question, sorted descending (noisiest first)."""
    scores = {q: asymm_calc(P) for q, P in p_matrices.items()}
    return pd.Series(scores, name="asymm").sort_values(ascending=False)


def plot_asymmetry_bar(asymm_series: pd.Series, ax=None, top_n: int | None = None):
    """Bar chart of asymm scores per item. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    series = asymm_series if top_n is None else asymm_series.head(top_n)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 5))
    series.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Asymmetry score (asymm)")
    ax.set_title("Most inconsistently-answered items in the survey (by P-matrix asymmetry)")
    return ax


def plot_p_matrix_heatmap(P: np.ndarray, title: str = "", ax=None):
    """Heatmap of a single 4x4 reliability matrix. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        P, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False,
        xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4],
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("response at 2nd measurement")
    ax.set_ylabel("response at 1st measurement")
    return ax
