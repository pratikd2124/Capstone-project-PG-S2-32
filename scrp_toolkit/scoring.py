"""Bulk-score real RY25 survey data for one case-study school.

Faithful port of the notebook's Section 2b. During training the network only ever saw
simulated data at the size of a single class/school (at most ~9,800-9,900 students), so we
score one representative school rather than the entire ~100k-student RY25 cohort, to stay
inside the model's training range.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .inference import UnreliabilityPredictor
from .reliability import asymm_calc


def pick_case_study_school(paths: ProjectPaths, p_matrices: Dict[str, np.ndarray]) -> tuple[str, list[str]]:
    """Return (school_id, question_list) for the school with the largest Pre/Post overlap."""
    if not paths.have_ry25_data:
        raise RuntimeError("RY25 Pre/Post CSVs were not found under the project root.")

    pre_cols = pd.read_csv(paths.pre_csv_path, nrows=0).columns.tolist()
    post_cols = pd.read_csv(paths.post_csv_path, nrows=0).columns.tolist()
    question_list = [q for q in p_matrices if q in pre_cols and q in post_cols and np.shape(p_matrices[q]) == (4, 4)]

    pre_df = pd.read_csv(paths.pre_csv_path, usecols=["school"] + question_list)
    post_df = pd.read_csv(paths.post_csv_path, usecols=["school"] + question_list)

    pre_sizes = pre_df.groupby("school").size()
    post_sizes = post_df.groupby("school").size()
    overlap = pd.DataFrame({"n_pre": pre_sizes, "n_post": post_sizes}).dropna()
    overlap["n_min"] = overlap[["n_pre", "n_post"]].min(axis=1)
    overlap = overlap.sort_values("n_min", ascending=False)

    school = overlap.index[0]
    return school, question_list


def score_school(predictor: UnreliabilityPredictor, paths: ProjectPaths, school: str | None = None) -> pd.DataFrame:
    """Score every item for ``school`` (or the best-overlap school if not given).

    Returns a DataFrame with one row per item: question, n_pre, n_post, shape_hat, rate_hat,
    gamma_mean (= shape_hat / rate_hat), asymm -- sorted by gamma_mean descending. Only
    4-option items are scored (EXP002 cannot score the 6-option chs items).

    shape_hat/rate_hat parameterise the Gamma distribution EXP002 predicts for RYA's test
    statistic d-hat between the Pre and Post groups (Technical Overview s2.4); gamma_mean is its
    mean. It was labelled "item unreliability" in the notebook -- it is not an item score.
    """
    if not paths.have_ry25_data:
        raise RuntimeError("RY25 Pre/Post CSVs were not found under the project root.")

    p_matrices = predictor.p_matrices
    if school is None:
        school, question_list = pick_case_study_school(paths, p_matrices)
    else:
        pre_cols = pd.read_csv(paths.pre_csv_path, nrows=0).columns.tolist()
        post_cols = pd.read_csv(paths.post_csv_path, nrows=0).columns.tolist()
        question_list = [q for q in p_matrices if q in pre_cols and q in post_cols and np.shape(p_matrices[q]) == (4, 4)]

    pre_df = pd.read_csv(paths.pre_csv_path, usecols=["school"] + question_list)
    post_df = pd.read_csv(paths.post_csv_path, usecols=["school"] + question_list)
    sub_pre = pre_df[pre_df["school"] == school]
    sub_post = post_df[post_df["school"] == school]

    rows = []
    for q in question_list:
        xc = sub_pre[q].dropna().astype(int)
        yc = sub_post[q].dropna().astype(int)
        if xc.empty or yc.empty:
            continue
        x_counts = np.array([np.sum(xc == k) for k in range(1, 5)])
        y_counts = np.array([np.sum(yc == k) for k in range(1, 5)])
        if x_counts.sum() == 0 or y_counts.sum() == 0:
            continue
        shape_hat, rate_hat = predictor.predict(x_counts, y_counts, q)
        rows.append({
            "question": q,
            "n_pre": int(x_counts.sum()),
            "n_post": int(y_counts.sum()),
            "shape_hat": shape_hat,
            "rate_hat": rate_hat,
            "gamma_mean": shape_hat / rate_hat,
            "asymm": asymm_calc(np.asarray(p_matrices[q])),
        })

    scores = pd.DataFrame(rows).sort_values("gamma_mean", ascending=False).reset_index(drop=True)
    scores.attrs["school"] = school
    return scores


def plot_scores_bar(scores: pd.DataFrame, ax=None):
    """Bar chart of the predicted Gamma mean per item. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
    ax.bar(scores["question"], scores["gamma_mean"], color=colors)
    ax.set_ylabel("Mean of predicted Gamma (shape / rate)")
    ax.set_title("Mean of the predicted d-hat Gamma per item, case-study school")
    return ax
