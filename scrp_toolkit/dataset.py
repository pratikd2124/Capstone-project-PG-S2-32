"""Build the from-scratch training dataset from the raw ``instance_data/*.csv`` files.

Port of RYA's ``net_dataset_init_B.py`` -> ``D_hat_Dataset`` -- the version ``perform_exp.py``
(and so EXP002) actually imports -- with the settings EXP002 was trained with:
``P_transform='all_P_asymm', B_transform_true=False, XY_symm_true=False,
include_XY_dist=True, include_skwurtoses_true=True, log_output_data_true=False,
two_sided_outlier_percentage=1``.

The one difference from ``net_dataset_init.py`` (which Murat's notebook followed) is the rate
rescaling ``rate <- 2 * rate / (1/n_X + 1/n_Y)`` applied right after ``dropna()``. It changes the
rate targets, the rate min/max stored in EXP002.json, and which rows the rate-quantile outlier
trim removes. ``normalise_rate=False`` reproduces the notebook instead.

Proof of equivalence: ``baseline`` rebuilds ``min_max_df`` from instance_data/ and compares all
84 values with the ``min_max_df`` stored in the customer's EXP002.json.
"""
from __future__ import annotations

import ast
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .reliability import asymm_calc


def _r_matrix_to_features(p_matrix_str: str) -> pd.Series:
    """Same as ``misc_functions.R_matrix_to_np``: parses the "c(0.1, 0.2, ...)" R-formatted vector."""
    py_list = "[" + p_matrix_str[2:-1] + "]"
    arr = np.array(ast.literal_eval(py_list))
    P = arr.reshape(4, 4).T
    d = {f"P_{i}": P.flat[i] for i in range(16)}
    d["asymm"] = asymm_calc(P)
    return pd.Series(d)


def build_preprocessed_dataset(
    instance_dir: str, two_sided_outlier_percentage: float = 1, normalise_rate: bool = True
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Load, feature-engineer, normalise and outlier-trim every CSV in ``instance_dir``.

    Returns (df, feature_cols, min_max_df):
        df           - the preprocessed, min-max normalised, outlier-trimmed dataframe
        feature_cols - the input feature column names (everything except exp_code/shape/rate)
        min_max_df   - the min/max used for normalisation of every column (needed to
                       reverse-normalise predictions, and to reproduce this exact scaling later)
    """
    csv_files = sorted(glob.glob(os.path.join(instance_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {instance_dir}.")

    df = pd.concat([pd.read_csv(f, index_col=False) for f in csv_files], ignore_index=True)
    df = df.dropna()

    if normalise_rate:
        # net_dataset_init_B.py: "Normalise rate parameter" (applied to orthog_B_1-4, raw rate)
        df["rate"] = 2 * df["rate"] / (1 / df["n_X"] + 1 / df["n_Y"])

    # P_matrix (R string) -> P_0..P_15 + asymm features
    p_feats = df.apply(lambda row: _r_matrix_to_features(row["P_matrix"]), axis=1)
    df = df.drop(["P_matrix", "P_matrix_question"], axis=1)
    i = df.columns.get_loc("shape")
    df = pd.concat([df.iloc[:, :i], p_feats, df.iloc[:, i:]], axis=1)

    # XY_symm_true = False -> keep only the first occurrence of each exp_code
    unique_idx = np.sort(np.unique(df["exp_code"], return_index=True)[1])
    df = df.iloc[unique_idx].reset_index(drop=True)

    # Min-max normalisation (all columns except exp_code) -- computed BEFORE outlier trimming
    non_normalise_cols = ["exp_code"]
    norm_cols = [c for c in df.columns if c not in non_normalise_cols]
    min_max_df = pd.DataFrame({
        "column": norm_cols,
        "min": [df[c].min() for c in norm_cols],
        "max": [df[c].max() for c in norm_cols],
    })
    for _, row in min_max_df.iterrows():
        col, mn, mx = row["column"], row["min"], row["max"]
        df[col] = (df[col] - mn) / (mx - mn)

    # Trim outliers: outer 1% (by default) separately for shape and rate
    sz = df["shape"].size - 1
    shape_q = df["shape"].rank(method="max").apply(lambda x: 100.0 * (x - 1) / sz)
    rate_q = df["rate"].rank(method="max").apply(lambda x: 100.0 * (x - 1) / sz)
    lower, upper = two_sided_outlier_percentage, 100 - two_sided_outlier_percentage
    keep = (shape_q > lower) & (shape_q < upper) & (rate_q > lower) & (rate_q < upper)
    df = df[keep].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["exp_code", "shape", "rate"]]
    return df, feature_cols, min_max_df


class DHatTensorDataset(Dataset):
    """Turns a pre-computed (normalised) df into a PyTorch Dataset."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.X = torch.tensor(df[feature_cols].values.astype(np.float32))
        self.y = torch.tensor(df[["shape", "rate"]].values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_test_split_like_original(
    df: pd.DataFrame, train_frac: float = 0.8, random_seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same logic as the ``np.random.choice``-based split in ``net_dataset_init.py``."""
    rng_state = np.random.get_state()
    np.random.seed(random_seed)
    n = len(df)
    n_train = round(train_frac * n)
    all_idx = np.arange(n)
    train_idx = np.random.choice(all_idx, size=n_train, replace=False)
    test_idx = np.array(sorted(set(all_idx) - set(train_idx)))
    np.random.set_state(rng_state)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
