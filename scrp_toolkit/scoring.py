"""Scoring every item for one real school in the RY25 survey.

This is how the model is actually used: take a school's answers before (Pre) and after
(Post), count how many students picked each answer for each item, and run the model on
every item.

We score one school rather than the whole ~100,000-student cohort because RYA only trained
the model on classroom/school-sized groups (up to about 9,900 students). By default we pick
the school with the most students in both surveys.

Only 4-option items are scored. The six 6-option chs items are skipped, because EXP002
can't handle them.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .inference import UnreliabilityPredictor
from .reliability import asymm_calc


def pick_case_study_school(paths: ProjectPaths, p_matrices: Dict[str, np.ndarray]) -> tuple[str, list[str]]:
    """The school with the most students in both surveys, and the items we can score for it."""
    if not paths.have_ry25_data:
        raise RuntimeError("RY25 Pre/Post CSVs were not found under the project root.")

    # Items that appear in both survey files and are 4-option.
    pre_cols = pd.read_csv(paths.pre_csv_path, nrows=0).columns.tolist()
    post_cols = pd.read_csv(paths.post_csv_path, nrows=0).columns.tolist()
    question_list = [q for q in p_matrices if q in pre_cols and q in post_cols and np.shape(p_matrices[q]) == (4, 4)]

    pre_df = pd.read_csv(paths.pre_csv_path, usecols=["school"] + question_list)
    post_df = pd.read_csv(paths.post_csv_path, usecols=["school"] + question_list)

    # Rank schools by their smaller group (before or after), so both groups are big.
    pre_sizes = pre_df.groupby("school").size()
    post_sizes = post_df.groupby("school").size()
    overlap = pd.DataFrame({"n_pre": pre_sizes, "n_post": post_sizes}).dropna()
    overlap["n_min"] = overlap[["n_pre", "n_post"]].min(axis=1)
    overlap = overlap.sort_values("n_min", ascending=False)

    school = overlap.index[0]
    return school, question_list


def score_school(predictor: UnreliabilityPredictor, paths: ProjectPaths, school: str | None = None) -> pd.DataFrame:
    """One row per item for ``school`` (or the best-covered school if none is given).

    Columns: question, n_pre, n_post, shape_hat, rate_hat, gamma_mean, asymm; sorted with the
    biggest gamma_mean first. The school id is kept in ``scores.attrs["school"]``.

    How to read gamma_mean (= shape / rate): the size of before/after difference that answer
    noise alone would typically produce for this item at these group sizes. It is *not* a
    reliability score for the item, which is what Murat's notebook called it.
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
        # Answers for this item, ignoring students who skipped it.
        xc = sub_pre[q].dropna().astype(int)
        yc = sub_post[q].dropna().astype(int)
        if xc.empty or yc.empty:
            continue
        # How many students picked 1, 2, 3 and 4.
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
    """Bar chart of gamma_mean for every scored item."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
    ax.bar(scores["question"], scores["gamma_mean"], color=colors)
    ax.set_ylabel("Mean of predicted Gamma (shape / rate)")
    ax.set_title("Mean of the predicted d-hat Gamma per item, case-study school")
    return ax
