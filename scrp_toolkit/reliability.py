"""The unreliability matrices, and how lopsided each one is.

Students don't always give the same answer when asked the same question twice. For each survey
item, RYA measured this as a matrix P: row j is how a student who "really" belongs in answer j
actually spreads their answers across 1..4. A perfectly reliable item would have 1s on the
diagonal and 0s elsewhere.

The ``asymm`` score summarises how lopsided the matrix is: whether students drift more one way
than the other. It's one of the model's 40 inputs. The maths is exactly RYA's ``asymm_calc``
from misc_functions.py (walkthrough Step 3 runs both side by side).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# The six pairs of cells above/below the diagonal of a 4x4 matrix (counting from 0).
_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def asymm_calc(P: np.ndarray) -> float:
    """How lopsided a matrix is: the average of (P[i, j] - P[j, i])^2 over the six pairs.

    0 means perfectly symmetric (students drift equally both ways); bigger means they tend to
    drift in one direction. Note: for the 6-option chs items this only looks at the top-left
    4x4 corner, just like RYA's version, so it isn't meaningful for them.
    """
    return sum((P[i, j] - P[j, i]) ** 2 for i, j in _PAIRS) / len(_PAIRS)


def load_p_matrices(pmatrices_path: str | Path) -> Dict[str, np.ndarray]:
    """Read unreliability-matrices.json into {item code: matrix}."""
    with open(pmatrices_path) as f:
        raw = json.load(f)
    return {q: np.array(P) for q, P in raw.items()}


def score_asymmetry(p_matrices: Dict[str, np.ndarray]) -> pd.Series:
    """The asymm score of every item, most lopsided first."""
    scores = {q: asymm_calc(P) for q, P in p_matrices.items()}
    return pd.Series(scores, name="asymm").sort_values(ascending=False)


def plot_asymmetry_bar(asymm_series: pd.Series, ax=None, top_n: int | None = None):
    """Bar chart of asymm scores, optionally just the top ``top_n`` items."""
    import matplotlib.pyplot as plt   # imported here so the rest of the module works without it

    series = asymm_series if top_n is None else asymm_series.head(top_n)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 5))
    series.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Asymmetry score (asymm)")
    ax.set_title("Most inconsistently-answered items in the survey (by P-matrix asymmetry)")
    return ax


def plot_p_matrix_heatmap(P: np.ndarray, title: str = "", ax=None):
    """Heatmap of one matrix, laid out like the images in Box's Heatmaps/NonTimeReversed folder."""
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
