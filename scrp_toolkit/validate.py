"""Check the ONNX model's predictions against a set of known answers.

This is the automated version of the notebook's "single example" check (Section 2, item
`ry9`): a small JSON file of {question, x, y, expected_shape, expected_rate} cases is scored
and compared against the model's output within a tolerance. Use this after any change to
the feature pipeline, the ONNX model file, or the metadata file, to confirm nothing broke.

`known_answers/ry9_example.json` ships the one worked example that was already in the
notebook. Add more cases there as they're confirmed against the original portal script /
Murat's comparative outputs.
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
    question: str
    x: List[int]
    y: List[int]
    expected_shape: float
    expected_rate: float
    label: str = ""


@dataclass
class ValidationResult:
    case: ValidationCase
    got_shape: float
    got_rate: float
    shape_ok: bool
    rate_ok: bool

    @property
    def passed(self) -> bool:
        return self.shape_ok and self.rate_ok


def load_known_answers(path: str | Path) -> List[ValidationCase]:
    with open(path) as f:
        raw = json.load(f)
    return [ValidationCase(**case) for case in raw]


def validate(
    predictor: UnreliabilityPredictor,
    cases: List[ValidationCase],
    rel_tol: float = 0.01,
) -> List[ValidationResult]:
    """Run every case through ``predictor`` and check it's within ``rel_tol`` relative tolerance."""
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
    """Print a pass/fail line per case. Returns True iff every case passed."""
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
