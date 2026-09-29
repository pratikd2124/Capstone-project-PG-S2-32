"""A quick check of the model against a few known answers.

This is the light-weight cousin of baseline.py: it runs a handful of cases from a small JSON
file (item, before counts, after counts, expected shape and rate) and says PASS or FAIL for
each. Handy as a 5-second sanity check after touching features.py or inference.py. For the
full comparison with RYA, run ``python -m scrp_toolkit.cli baseline``.

known_answers/ry9_example.json holds RYA's own worked example (item ry9).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .inference import UnreliabilityPredictor


@dataclass
class ValidationCase:
    """One known answer: the inputs and what the model should return."""

    question: str
    x: List[int]
    y: List[int]
    expected_shape: float
    expected_rate: float
    label: str = ""


@dataclass
class ValidationResult:
    """What the model actually returned for one case, and whether it was close enough."""

    case: ValidationCase
    got_shape: float
    got_rate: float
    shape_ok: bool
    rate_ok: bool

    @property
    def passed(self) -> bool:
        return self.shape_ok and self.rate_ok


def load_known_answers(path: str | Path) -> List[ValidationCase]:
    """Read the known-answer cases from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return [ValidationCase(**case) for case in raw]


def validate(
    predictor: UnreliabilityPredictor,
    cases: List[ValidationCase],
    rel_tol: float = 0.01,
) -> List[ValidationResult]:
    """Run each case through the model; "close enough" means within ``rel_tol`` (1%) of expected."""
    results = []
    for case in cases:
        got_shape, got_rate = predictor.predict(np.array(case.x), np.array(case.y), case.question)
        results.append(
            ValidationResult(
                case=case,
                got_shape=got_shape,
                got_rate=got_rate,
                shape_ok=np.isclose(got_shape, case.expected_shape, rtol=rel_tol),
                rate_ok=np.isclose(got_rate, case.expected_rate, rtol=rel_tol),
            )
        )
    return results


def print_report(results: List[ValidationResult]) -> bool:
    """Print one PASS/FAIL line per case. Returns True only if every case passed."""
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        all_passed &= r.passed
        label = r.case.label or r.case.question
        print(
            f"[{status}] {label}: shape={r.got_shape:.4f} (expected {r.case.expected_shape:.4f}), "
            f"rate={r.got_rate:.4f} (expected {r.case.expected_rate:.4f})"
        )
    print(f"\n{sum(r.passed for r in results)}/{len(results)} cases passed.")
    return all_passed
