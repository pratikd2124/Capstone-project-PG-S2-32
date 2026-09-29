"""Rebuilding RYA's training data from the raw CSV files.

The training data is four CSV files (orthog_B_1..4) in instance_data/. Each row is one
simulated classroom comparison: the 23 answer-based features, the item's matrix (as text),
and the "true" Gamma shape and rate the model had to learn.

RYA prepared these rows in net_dataset_init_B.py (the version perform_exp.py imports, so the
one EXP002 was trained with). This module does exactly the same, in the same order:

    1. drop rows with missing values (groups where everyone gave the same answer)
    2. rescale rate by 2 / (1/n_X + 1/n_Y)          <- the step Murat's notebook left out
    3. unpack the matrix text into P_0..P_15 and asymm
    4. keep the first copy of each exp_code
    5. squeeze every column into 0-1 (min-max), and remember the min/max
    6. drop the outer 1% of shape and of rate
    7. (separately) split 80/20 into train and test

Proof it matches: the min/max from step 5 reproduce all 84 values stored in EXP002.json
(walkthrough Step 6, and the "scaling" section of the baseline). With normalise_rate=False you
get Murat's notebook version instead, and the rate min/max no longer match.
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
    """Turn R's "c(0.1, 0.2, ...)" matrix text into P_0..P_15 + asymm (RYA's R_matrix_to_np).

    R writes matrices column by column, so the flat list is reshaped and then transposed.
    """
    py_list = "[" + p_matrix_str[2:-1] + "]"
    arr = np.array(ast.literal_eval(py_list))
    P = arr.reshape(4, 4).T
    d = {f"P_{i}": P.flat[i] for i in range(16)}
    d["asymm"] = asymm_calc(P)
    return pd.Series(d)


def build_preprocessed_dataset(
    instance_dir: str, two_sided_outlier_percentage: float = 1, normalise_rate: bool = True
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Load every CSV in ``instance_dir`` and prepare it exactly as RYA did.

    Returns three things:
        df            the prepared rows, every column scaled to 0-1
        feature_cols  the 40 input column names, in the order the model expects
        min_max_df    the min and max used to scale each column (compare with EXP002.json)
    """
    csv_files = sorted(glob.glob(os.path.join(instance_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {instance_dir}.")

    df = pd.concat([pd.read_csv(f, index_col=False) for f in csv_files], ignore_index=True)

    # 1. Rows with NaN come from groups where everyone gave the same answer (68 rows).
    df = df.dropna()

    # 2. RYA's rate rescaling. The CSVs hold the raw rate; this turns it into the rate of the
    #    d-hat statistic including its 1/2 (1/n + 1/m) factor, which is what EXP002 learnt.
    if normalise_rate:
        df["rate"] = 2 * df["rate"] / (1 / df["n_X"] + 1 / df["n_Y"])

    # 3. The matrix text becomes 17 numeric columns, placed just before shape/rate.
    p_feats = df.apply(lambda row: _r_matrix_to_features(row["P_matrix"]), axis=1)
    df = df.drop(["P_matrix", "P_matrix_question"], axis=1)
    i = df.columns.get_loc("shape")
    df = pd.concat([df.iloc[:, :i], p_feats, df.iloc[:, i:]], axis=1)

    # 4. Keep only the first row for each exp_code (the data can hold mirrored copies).
    unique_idx = np.sort(np.unique(df["exp_code"], return_index=True)[1])
    df = df.iloc[unique_idx].reset_index(drop=True)

    # 5. Scale every column to 0-1. The min/max are taken BEFORE the outlier trim, as RYA did.
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

    # 6. Drop the most extreme 1% at each end, for shape and for rate separately.
    sz = df["shape"].size - 1
    shape_q = df["shape"].rank(method="max").apply(lambda x: 100.0 * (x - 1) / sz)
    rate_q = df["rate"].rank(method="max").apply(lambda x: 100.0 * (x - 1) / sz)
    lower, upper = two_sided_outlier_percentage, 100 - two_sided_outlier_percentage
    keep = (shape_q > lower) & (shape_q < upper) & (rate_q > lower) & (rate_q < upper)
    df = df[keep].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["exp_code", "shape", "rate"]]
    return df, feature_cols, min_max_df


class DHatTensorDataset(Dataset):
    """Wraps the prepared rows so PyTorch can feed them to the network in batches."""

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
    """Split into train and test the way RYA did (a random 80% for training, the rest for test).

    The same seed always gives the same split. Python's global random state is put back
    afterwards, so calling this doesn't disturb anything else that uses randomness.
    """
    rng_state = np.random.get_state()
    np.random.seed(random_seed)
    n = len(df)
    n_train = round(train_frac * n)
    all_idx = np.arange(n)
    train_idx = np.random.choice(all_idx, size=n_train, replace=False)
    test_idx = np.array(sorted(set(all_idx) - set(train_idx)))
    np.random.set_state(rng_state)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
