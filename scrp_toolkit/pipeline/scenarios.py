"""The demo cases: made-up comparisons that each show one part of the pipeline at work.

They are for the demonstration, not evidence. One case (J) is a real instance from
RYA's training data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ComparisonRequest
from .simulate import integer_composition


@dataclass
class Scenario:
    name: str
    objective: str      # which part of the pipeline it shows
    expect: str         # what should happen
    request: ComparisonRequest


def counts(shares, n):
    """Whole-student counts for a group of n with roughly these answer shares."""
    return integer_composition(np.asarray(shares, float), int(n))


def training_row(exp_code, instance_dir):
    """One instance from RYA's training data, with its own rho."""
    for path in Path(instance_dir).glob("*.csv"):
        df = pd.read_csv(path)
        match = df[df["exp_code"] == exp_code]
        if len(match):
            row = match.iloc[0]
            numbers = [float(v) for v in re.findall(r"[-\d.]+", row["P_matrix"])]
            return {"x": [row[f"X_{i}"] for i in range(1, 5)], "y": [row[f"Y_{i}"] for i in range(1, 5)],
                    "n": int(row["n_X"]), "m": int(row["n_Y"]), "item": row["P_matrix_question"],
                    "P": np.array(numbers).reshape(4, 4, order="F")}    # R stores matrices column by column
    raise KeyError(exp_code)


def synthetic_transition_counts(P, per_row, seed=11):
    """Made-up test-retest data: per_row students of each type answering through rho."""
    rng = np.random.default_rng(seed)
    P = np.asarray(P) / np.asarray(P).sum(axis=1, keepdims=True)
    return np.stack([rng.multinomial(per_row, P[j]) for j in range(4)]).astype(float)


def build_scenarios(p_matrices, instance_dir=None, epsilon=0.1):
    rho = p_matrices["ry9"]
    national = counts([0.20, 0.32, 0.28, 0.20], 9000)       # a national cohort answering ry9
    cases = []

    def add(name, objective, expect, item, x, y, P, label, **options):
        cases.append(Scenario(name, objective, expect,
                              ComparisonRequest.make(item, x, y, P, epsilon=epsilon, label=label, **options)))

    add("A. Typical comparison", "O4 network + O3 range", "passes screening, uses the network, GREEN",
        "ry9", [40, 52, 7, 14], [9, 17, 54, 12], rho, "A typical")
    add("B. Classroom (n=8) vs national", "O1 screening + O2 fallback",
        "flagged, uses the fallback; the network alone would wrongly call a meaningful difference",
        "ry9", [2, 1, 2, 3], national, rho, "B classroom n=8")
    add("C. Classroom (n=25) vs national", "O1 screening",
        "the classifier passes it; the small-group rule sends it to the fallback",
        "ry9", counts([0.1, 0.25, 0.4, 0.25], 25), national, rho, "C classroom n=25")
    add("D. Everyone gave one answer", "O1 degeneracy safeguard", "flagged, uses the fallback",
        "ry9", [0, 0, 12, 0], [30, 41, 25, 14], rho, "D degenerate")

    broken = np.array(rho, float)
    broken[0] *= 1.06
    add("E. Corrupt rho (row sums 1.06)", "O1 rho check", "rejected, RED, no p-value",
        "ry9", [40, 52, 7, 14], [9, 17, 54, 12], broken, "E bad rho")
    add("F. Six-option item", "O1 scope check", "rejected, RED",
        "chs1", [3, 4, 5, 6], [6, 5, 4, 3], np.eye(4), "F chs1")
    add("G. Cohort above training range (n=15,000)", "O1 training-range check", "flagged, uses the fallback",
        "ry9", counts([0.18, 0.33, 0.29, 0.20], 15000), national, rho, "G n=15000")

    group_x, group_y = [24, 33, 24, 19], [14, 30, 34, 22]
    add("H. Rho from a small retest study (15/row, SYNTHETIC)", "O3 rho uncertainty",
        "wide p-range that crosses 0.05, AMBER", "ry9", group_x, group_y, rho, "H rho 15/row",
        rho_transition_counts=synthetic_transition_counts(rho, per_row=15))
    add("I. Same, large retest study (2,000/row, SYNTHETIC)", "O3 rho uncertainty",
        "narrow p-range, GREEN", "ry9", group_x, group_y, rho, "I rho 2000/row",
        rho_transition_counts=synthetic_transition_counts(rho, per_row=2000))

    if instance_dir is not None:
        try:
            row = training_row("orthog_B_3_large_classes_2166", instance_dir)
        except (KeyError, FileNotFoundError):
            row = None
        if row:
            add("J. RYA training instance n=6 vs 5,215 (ry24)", "O1 acceptance case (SIN-WEI)",
                "flagged by the classifier, uses the fallback", row["item"],
                counts(row["x"], row["n"]), counts(row["y"], row["m"]), row["P"], "J ry24 n=6",
                rho_source="instance")
    return cases
