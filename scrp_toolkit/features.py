"""Feature engineering shared between inference and training.

Faithful port of ``feature_calculation_portal_script.py`` (the notebook's Section 2), kept
independent of the ONNX runtime so it can be unit-tested and reused by the training pipeline.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .reliability import asymm_calc


def moment_calculator(prob_dist: np.ndarray, dist_name: str, num_types: int = 4) -> pd.Series:
    """Mean, std, skewness, kurtosis of a Pre/Post response distribution."""
    rv_vec = np.arange(1, num_types + 1)
    mean = np.dot(rv_vec, prob_dist)
    stdev = np.sqrt(np.dot((rv_vec - mean) ** 2, prob_dist))
    skewness = np.dot((rv_vec - mean) ** 3, prob_dist) / stdev ** 3
    kurtosis = np.dot((rv_vec - mean) ** 4, prob_dist) / stdev ** 4
    return pd.Series({
        f"mean_{dist_name}": mean,
        f"stdev_{dist_name}": stdev,
        f"skewness_{dist_name}": skewness,
        f"kurtosis_{dist_name}": kurtosis,
    })


def transform_p_matrix(question: str, p_matrices: Dict[str, np.ndarray]) -> pd.Series:
    """Flatten a question's 4x4 P-matrix into P_0..P_15 plus its asymm score."""
    P = np.asarray(p_matrices[question])
    d = {f"P_{i}": P.flat[i] for i in range(P.size)}
    d["asymm"] = asymm_calc(P)
    return pd.Series(d)


def extract_features(x: np.ndarray, y: np.ndarray, question: str, p_matrices: Dict[str, np.ndarray]) -> pd.Series:
    """Build the full feature vector for one item's Pre (x) / Post (y) response counts.

    x, y: length-4 arrays of response counts for options 1..4.
    question: item code (must be a key in ``p_matrices``), e.g. 'ry9'.

    Raises ValueError for anything other than a 4-option item. EXP002 was trained on 4x4
    unreliability matrices only (P_0..P_15); the customer's matrices file also holds 6-option
    items (chs1-chs6, 6x6). Fed through unchecked, those silently produce a scrambled
    P_0..P_15 and drop answers 5-6 -- a meaningless score, not an error.
    """
    P = np.asarray(p_matrices[question])
    if P.shape != (4, 4):
        raise ValueError(f"Item '{question}' has a {P.shape[0]}x{P.shape[1]} unreliability matrix; "
                         "the EXP002 network only supports 4-option items (4x4).")
    if np.size(x) != 4 or np.size(y) != 4:
        raise ValueError(f"Expected 4 response counts for x and y, got {np.size(x)} and {np.size(y)}.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_X, n_Y = x.sum(), y.sum()
    X, Y = x / n_X, y / n_Y
    XmY = np.abs(X - Y)
    euclid_dist_XY = np.round(np.sqrt(np.sum((X - Y) ** 2)), 4)

    parts = [
        pd.Series({f"X_{i + 1}": X[i] for i in range(4)}),
        pd.Series({f"Y_{i + 1}": Y[i] for i in range(4)}),
        pd.Series({f"XmY_{i + 1}": XmY[i] for i in range(4)}),
        moment_calculator(X, "X"),
        moment_calculator(Y, "Y"),
        pd.Series({"euclid_dist_XY": euclid_dist_XY}),
        pd.Series({"n_X": n_X, "n_Y": n_Y}),
        transform_p_matrix(question, p_matrices),
    ]
    return pd.concat(parts)
