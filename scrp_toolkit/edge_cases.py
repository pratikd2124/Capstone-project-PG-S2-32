"""Generate edge-case data for stress-testing the pipeline and training a "is this safe?" detector.

Two kinds of edge cases are produced, matching the two places bad input can hit the pipeline:

1. **Response-count edge cases** (``generate_response_count_cases``): unusual Pre/Post count
   pairs (x, y) fed straight into ``features.extract_features`` / ``inference.predict``, e.g. a
   single respondent, everyone picking the same option, a category nobody picked at all, or a
   complete reversal between Pre and Post. These are for regression/robustness testing of the
   feature pipeline itself (does it divide by zero, does it silently produce NaN/inf, does it
   crash) -- run them through ``scrp_toolkit.cli validate`` or the pipeline directly.

2. **Instance-data-schema edge cases** (``generate_instance_data_rows``): synthetic rows in the
   same shape as the real ``instance_data/*.csv`` files (P_matrix, P_matrix_question, exp_code,
   shape, rate, plus the raw X/Y features), but with deliberately extreme or degenerate
   P-matrices (identity, uniform/maximally noisy, near-singular, fully asymmetric). These are
   labelled bad-input examples for SIN-WEI's "is this safe?" detector (labels are written to a
   separate file, keyed by ``exp_code``). They are NOT training data for the regressor -- their
   shape/rate values are placeholders.

Every case is deterministic given a seed, so the same edge-case set can be regenerated and
diffed later rather than silently drifting between runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Response-count edge cases (x, y, question) -> feed straight into the pipeline
# ---------------------------------------------------------------------------


@dataclass
class ResponseCountCase:
    label: str
    description: str
    x: List[int]
    y: List[int]
    question: str = "ry9"
    expect_failure: bool = False  # True = this case is expected to raise / be rejected, not silently score


def generate_response_count_cases(question: str = "ry9") -> List[ResponseCountCase]:
    """A fixed catalogue of edge cases for Pre/Post response-count pairs.

    Covers: zero-variance responses, a single respondent, an unpicked category, complete
    reversal between measurements, wildly mismatched Pre/Post sample sizes, and the
    genuinely-invalid case of an empty count vector.
    """
    return [
        ResponseCountCase(
            label="all_same_option",
            description="Every student picks option 1 at both measurements (zero variance -- perfectly 'reliable' by construction).",
            x=[100, 0, 0, 0], y=[100, 0, 0, 0], question=question,
        ),
        ResponseCountCase(
            label="single_respondent",
            description="Only one student answered (n=1) -- the smallest non-empty group the pipeline should handle.",
            x=[1, 0, 0, 0], y=[0, 0, 0, 1], question=question,
        ),
        ResponseCountCase(
            label="unpicked_category",
            description="Option 3 is never picked at either measurement -- tests that a zero-count column doesn't break moment/skew calculations.",
            x=[30, 40, 0, 30], y=[20, 50, 0, 30], question=question,
        ),
        ResponseCountCase(
            label="complete_reversal",
            description="Everyone who picked option 1 at Pre picks option 4 at Post, and vice versa -- maximum possible instability.",
            x=[80, 10, 5, 5], y=[5, 5, 10, 80], question=question,
        ),
        ResponseCountCase(
            label="mismatched_sample_sizes",
            description="Pre has 10x the respondents of Post (e.g. a student left the school between surveys) -- tests n_X/n_Y handling, not just proportions.",
            x=[250, 250, 250, 250], y=[10, 10, 5, 5], question=question,
        ),
        ResponseCountCase(
            label="extreme_skew",
            description="Almost everyone picks the extreme option at Pre, almost everyone picks the opposite extreme at Post.",
            x=[199, 1, 0, 0], y=[0, 0, 1, 199], question=question,
        ),
        ResponseCountCase(
            label="empty_group",
            description="No respondents at all -- INVALID input (n_X=0 -> division by zero). The pipeline should raise or reject this, not silently return a number.",
            x=[0, 0, 0, 0], y=[0, 0, 0, 0], question=question, expect_failure=True,
        ),
        ResponseCountCase(
            label="unknown_question_code",
            description="A question code that doesn't exist in unreliability-matrices.json -- INVALID input, should raise a clear KeyError, not silently produce garbage features.",
            x=[25, 25, 25, 25], y=[25, 25, 25, 25], question="__not_a_real_item__", expect_failure=True,
        ),
    ]


def check_response_count_cases(cases: List[ResponseCountCase], p_matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Run every case through ``features.extract_features`` and report what actually happened.

    This is the "safety net" check: does the case behave the way ``expect_failure`` says it
    should? A row where ``expect_failure`` doesn't match ``raised`` is worth a look -- either the
    pipeline needs an explicit guard (for the empty-group / unknown-question cases), or a case
    marked ``expect_failure=True`` is stricter than it needs to be.
    """
    from .features import extract_features

    rows = []
    for case in cases:
        raised, error, has_nan, has_inf = False, "", False, False
        try:
            feats = extract_features(np.array(case.x), np.array(case.y), case.question, p_matrices)
            has_nan = bool(np.isnan(feats.values.astype(float)).any())
            has_inf = bool(np.isinf(feats.values.astype(float)).any())
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, we're cataloguing *any* failure
            raised = True
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "label": case.label,
            "expect_failure": case.expect_failure,
            "raised": raised,
            "produced_nan": has_nan,
            "produced_inf": has_inf,
            "matches_expectation": raised == case.expect_failure and not has_nan and not has_inf,
            "error": error,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Instance-data-schema edge cases -> synthetic rows for the "is this safe?" detector
# ---------------------------------------------------------------------------

_R_ROW_ORDER = "row-major, matches misc_functions.R_matrix_to_np (P.reshape(4,4).T)"


def _matrix_to_r_string(P: np.ndarray) -> str:
    """Inverse of dataset._r_matrix_to_features: turn a 4x4 array into an R-style 'c(...)' string.

    dataset._r_matrix_to_features does ``P = arr.reshape(4, 4).T``, so to round-trip a matrix P
    back to the string that would reproduce it, we transpose before flattening.
    """
    flat = P.T.flatten()
    return "c(" + ", ".join(f"{v:.6f}" for v in flat) + ")"


@dataclass
class EdgeCaseMatrixSpec:
    label: str
    description: str
    build: "callable"  # () -> np.ndarray, a 4x4 row-stochastic matrix


def _identity_like(diag: float = 0.97) -> np.ndarray:
    off = (1 - diag) / 3
    P = np.full((4, 4), off)
    np.fill_diagonal(P, diag)
    return P


def _uniform_noise() -> np.ndarray:
    return np.full((4, 4), 0.25)


def _fully_asymmetric() -> np.ndarray:
    # Deterministic hand-built matrix maximising the off-diagonal asymmetry pairs
    return np.array([
        [0.10, 0.60, 0.20, 0.10],
        [0.05, 0.10, 0.70, 0.15],
        [0.60, 0.10, 0.10, 0.20],
        [0.15, 0.15, 0.60, 0.10],
    ])


def _near_singular(rng: np.random.Generator) -> np.ndarray:
    # Two rows almost identical -> near rank-deficient, a known numerically awkward case
    base = rng.dirichlet(np.ones(4))
    P = np.tile(base, (4, 1))
    P += rng.normal(scale=1e-4, size=(4, 4))
    P = np.clip(P, 1e-6, None)
    return P / P.sum(axis=1, keepdims=True)


_EDGE_MATRIX_SPECS: List[EdgeCaseMatrixSpec] = [
    EdgeCaseMatrixSpec(
        "near_perfect_reliability", "Diagonal-dominant matrix (0.97) -- an implausibly, almost suspiciously reliable item.",
        lambda rng: _identity_like(0.97),
    ),
    EdgeCaseMatrixSpec(
        "maximally_noisy", "Uniform 0.25 everywhere -- a completely random / unreliable item, the theoretical worst case.",
        lambda rng: _uniform_noise(),
    ),
    EdgeCaseMatrixSpec(
        "fully_asymmetric", "Hand-built matrix maximising off-diagonal asymmetry -- systematic directional drift, not just noise.",
        lambda rng: _fully_asymmetric(),
    ),
    EdgeCaseMatrixSpec(
        "near_singular", "Rows nearly identical (near rank-deficient) -- numerically awkward, can destabilise downstream linear algebra.",
        lambda rng: _near_singular(rng),
    ),
]


def generate_instance_data_rows(n_per_case: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize ``instance_data``-schema rows built around deliberately extreme P-matrices.

    Returns ``(rows, labels)``:

    * ``rows`` has exactly the columns of the real ``instance_data/*.csv`` files -- every raw
      feature ``features.extract_features`` produces (except the P_*/asymm ones, which
      ``dataset.build_preprocessed_dataset`` re-derives from ``P_matrix``), plus P_matrix,
      P_matrix_question, exp_code, shape, rate. No extra columns: an extra column would be NaN
      for every real row after a concat, and ``build_preprocessed_dataset`` calls ``dropna()``,
      which would then silently delete the entire real dataset.
    * ``labels`` maps each ``exp_code`` to its scenario, for use as the detector's label column.

    ``exp_code`` values are strings (``edge_<label>_<n>``), matching the real data's
    ``orthog_B_1_1``-style codes -- mixing int and str codes makes the dedup step's
    ``np.unique`` raise.

    ``shape``/``rate`` here are placeholders, not true unreliability values. Do NOT put these
    rows in ``instance_data/`` for regressor training -- they'd corrupt its targets. They're for
    the bad-input detector, trained on real rows (label 0) + these rows (label 1).
    """
    from .features import extract_features

    rng = np.random.default_rng(seed)
    rows, labels = [], []

    for spec in _EDGE_MATRIX_SPECS:
        for n in range(n_per_case):
            P = spec.build(rng)
            x = rng.integers(1, 100, size=4)
            y = rng.integers(1, 100, size=4)
            feats = extract_features(x, y, "ry9", {"ry9": P})
            feats = feats.drop([c for c in feats.index if c.startswith("P_")] + ["asymm"])
            exp_code = f"edge_{spec.label}_{n}"
            row = feats.to_dict()
            row.update({
                "exp_code": exp_code,
                "P_matrix": _matrix_to_r_string(P),
                "P_matrix_question": "ry9",
                "shape": float(rng.uniform(0.5, 3.0)),
                "rate": float(rng.uniform(0.5, 3.0)),
            })
            rows.append(row)
            labels.append({"exp_code": exp_code, "is_edge_case": 1, "edge_case_label": spec.label})

    return pd.DataFrame(rows), pd.DataFrame(labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate edge-case data for the SCRP pipeline (response-count cases and/or instance_data-schema rows)."
    )
    parser.add_argument("--project-root", default=None, help="Needed only for --check-response-cases (reads unreliability-matrices.json)")
    parser.add_argument("--out-dir", default="edge_cases", help="Folder to write the generated files into")
    parser.add_argument("--n-per-case", type=int, default=20, help="Rows to generate per instance-data edge-case matrix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-response-cases", action="store_true", help="Run the response-count cases through extract_features and report pass/fail")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    response_cases = generate_response_count_cases()
    response_path = out_dir / "response_count_cases.json"
    response_path.write_text(json.dumps([asdict(c) for c in response_cases], indent=2))
    print(f"Wrote {len(response_cases)} response-count edge cases: {response_path}")

    instance_rows, labels = generate_instance_data_rows(n_per_case=args.n_per_case, seed=args.seed)
    instance_path = out_dir / "edge_case_instance_data.csv"
    labels_path = out_dir / "edge_case_labels.csv"
    instance_rows.to_csv(instance_path, index=False)
    labels.to_csv(labels_path, index=False)
    print(f"Wrote {len(instance_rows)} instance_data-schema edge-case rows: {instance_path}")
    print(f"Wrote their labels (exp_code -> scenario): {labels_path}")

    if args.check_response_cases:
        if not args.project_root:
            parser.error("--check-response-cases requires --project-root (to load unreliability-matrices.json)")
        from .config import resolve_project_paths
        from .reliability import load_p_matrices

        paths = resolve_project_paths(args.project_root)
        p_matrices = load_p_matrices(paths.pmatrices_path)
        report = check_response_count_cases(response_cases, p_matrices)
        report_path = out_dir / "response_count_case_report.csv"
        report.to_csv(report_path, index=False)
        print(f"\n{report.to_string(index=False)}")
        print(f"\nSaved: {report_path}")
        if not report["matches_expectation"].all():
            print("\nSome cases did not match their expectation -- see the 'matches_expectation' column above.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
