"""Synthetic demonstration cases: one or more per objective.

These are built for the prototype demonstration, not as production evidence. Each case
says which part of the pipeline it exercises and what should happen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .contracts import ComparisonRequest as R


@dataclass
class Scenario:
    name: str
    objective: str
    expect: str
    request: R


def _counts(props, n):
    from .simulate import integer_composition
    return integer_composition(np.asarray(props, float), int(n))


def training_row(exp_code: str, instance_dir: str | Path) -> Dict:
    """One RYA training instance: shares, group sizes and its instance-specific rho."""
    for f in Path(instance_dir).glob("*.csv"):
        df = pd.read_csv(f)
        hit = df[df["exp_code"] == exp_code]
        if len(hit):
            row = hit.iloc[0]
            vals = [float(v) for v in re.findall(r"[-\d.]+", row["P_matrix"])]
            P = np.array(vals).reshape(4, 4, order="F")      # R stores matrices column by column
            return {"x": [row[f"X_{i}"] for i in range(1, 5)], "y": [row[f"Y_{i}"] for i in range(1, 5)],
                    "n": int(row["n_X"]), "m": int(row["n_Y"]), "P": P, "item": row["P_matrix_question"]}
    raise KeyError(exp_code)


def synthetic_transition_counts(P: np.ndarray, per_row: int, seed: int = 11) -> np.ndarray:
    """Pretend test-retest data: per_row students of each type, answering by rho (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    P = np.asarray(P) / np.asarray(P).sum(axis=1, keepdims=True)
    return np.stack([rng.multinomial(per_row, P[j]) for j in range(4)]).astype(float)


def build_scenarios(p_matrices: Dict[str, np.ndarray], instance_dir: str | Path | None = None,
                    epsilon: float = 0.1) -> List[Scenario]:
    P9 = p_matrices["ry9"]
    national = _counts([0.20, 0.32, 0.28, 0.20], 9000)   # a national cohort on ry9
    s: List[Scenario] = []

    s.append(Scenario("A. Typical comparison", "O4 network + O3 range",
                      "PASS -> network Gamma -> GREEN",
                      R.make("ry9", [40, 52, 7, 14], [9, 17, 54, 12], P9, epsilon=epsilon, label="A typical")))
    s.append(Scenario("B. Classroom (n=8) vs national", "O1 screening + O2 fallback",
                      "FLAGGED -> simulated null -> AMBER; network alone calls a false 'meaningful difference'",
                      R.make("ry9", [2, 1, 2, 3], national, P9, epsilon=epsilon, label="B classroom n=8")))
    s.append(Scenario("C. Classroom (n=25) vs national", "O1 screening",
                      "KNOWN GAP: screening passes, network p too small (feedback to O1)",
                      R.make("ry9", _counts([0.1, 0.25, 0.4, 0.25], 25), national, P9, epsilon=epsilon,
                             label="C classroom n=25")))
    s.append(Scenario("D. Everyone gave one answer", "O1 degeneracy safeguard",
                      "FLAGGED (degenerate) -> fallback -> AMBER",
                      R.make("ry9", [0, 0, 12, 0], [30, 41, 25, 14], P9, epsilon=epsilon, label="D degenerate")))
    bad = np.array(P9, float).copy()
    bad[0] = bad[0] * 1.06
    s.append(Scenario("E. Corrupt rho (row sums 1.06)", "O1 rho validation",
                      "REJECT -> RED, no p-value",
                      R.make("ry9", [40, 52, 7, 14], [9, 17, 54, 12], bad, epsilon=epsilon, label="E bad rho")))
    s.append(Scenario("F. Six-option item", "O1 scope check",
                      "REJECT -> RED", R.make("chs1", [3, 4, 5, 6], [6, 5, 4, 3], np.eye(4), epsilon=epsilon,
                                             label="F chs1")))
    s.append(Scenario("G. Cohort above training range (n=15,000)", "O1 training-range check",
                      "FLAGGED (outside 5-9,940) -> fallback",
                      R.make("ry9", _counts([0.18, 0.33, 0.29, 0.20], 15000), national, P9, epsilon=epsilon,
                             label="G n=15000")))
    tc_small = synthetic_transition_counts(P9, per_row=15)
    tc_large = synthetic_transition_counts(P9, per_row=2000)
    near = [24, 33, 24, 19]
    s.append(Scenario("H. Rho from a small retest study (15/row, SYNTHETIC)", "O3 rho uncertainty",
                      "wide p-range; AMBER if the decision flips",
                      R.make("ry9", near, [14, 30, 34, 22], P9, epsilon=epsilon, label="H rho 15/row",
                             rho_transition_counts=tc_small)))
    s.append(Scenario("I. Same, large retest study (2,000/row, SYNTHETIC)", "O3 rho uncertainty",
                      "narrow p-range; GREEN",
                      R.make("ry9", near, [14, 30, 34, 22], P9, epsilon=epsilon, label="I rho 2000/row",
                             rho_transition_counts=tc_large)))
    if instance_dir is not None:
        try:
            tr = training_row("orthog_B_3_large_classes_2166", instance_dir)
            s.append(Scenario("J. RYA training instance n=6 vs 5,215 (ry24)", "O1 acceptance case (SIN-WEI)",
                              "FLAGGED by classifier -> fallback",
                              R.make(tr["item"], _counts(tr["x"], tr["n"]), _counts(tr["y"], tr["m"]), tr["P"],
                                     epsilon=epsilon, label="J ry24 n=6", rho_source="instance")))
        except (KeyError, FileNotFoundError):
            pass
    return s
