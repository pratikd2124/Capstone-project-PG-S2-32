"""Steps 1a and 1b: check rho, then decide whether the network can be trusted (Objective 1).

The checks themselves are SIN-WEI's (o1_screening/o1_screening.py, unchanged). Three things are
added here:

* her detector runs from detector_params.json, the same numbers copied out of her joblib
  file, so it doesn't depend on the scikit-learn version that saved it;
* her classifier looks at the observed pair and at the boundary pair (the one the network
  actually sees), and either one can flag the request;
* two extra rules: group sizes outside the network's training range (5-9,940), and a small
  group (under 30) compared with one at least 20 times bigger. The second was added after
  testing showed the classifier passing a class of 25 against 9,000 when the network's
  answer was about 40% off.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import FLAGGED, PASS, REJECT, RhoCheck, ScreeningResult
from o1_screening.o1_screening import O1ScreeningGate, calculate_P_structure_descriptors

O1_DIR = Path(__file__).resolve().parent.parent / "o1_screening"
TRAINING_SIZES = (5, 9940)        # smallest and largest group the network was trained on
ROW_SUM_LIMIT = 0.020001          # SIN-WEI's limit for how far a rho row may be from 1
SMALL_GROUP_N = 30
SMALL_GROUP_RATIO = 20.0
SIX_OPTION_ITEMS = {f"chs{i}" for i in range(1, 7)}


class Gate(O1ScreeningGate):
    """SIN-WEI's gate, with the detector read from JSON instead of joblib."""

    def __post_init__(self):
        params = json.loads(Path(self.detector_model_path).read_text())
        self.detector = _LogisticDetector(params)
        self.detector_metadata = json.loads(Path(self.detector_metadata_path).read_text())
        self.feature_order = list(self.detector_metadata["feature_order"])
        self.threshold = float(self.detector_metadata["threshold"])
        self._p_scaler = self._p_nn = self._p_reference_df = self._training_nearest_distances = None
        if self.screening_reference_csv is not None:
            self._prepare_training_space_reference(Path(self.screening_reference_csv))


class _LogisticDetector:
    """Standard scaling followed by logistic regression: what her joblib file contains."""

    def __init__(self, params):
        self.mean = np.array(params["scaler_mean"])
        self.scale = np.array(params["scaler_scale"])
        self.coef = np.array(params["coef"])
        self.intercept = params["intercept"]

    def predict_proba(self, X):
        z = (np.asarray(X, dtype=float) - self.mean) / self.scale
        p = 1 / (1 + np.exp(-(z @ self.coef + self.intercept)))
        return np.column_stack([1 - p, p])


def load_gate(reference_csv=None, threshold=None):
    gate = Gate(detector_model_path=O1_DIR / "detector_params.json",
                detector_metadata_path=O1_DIR / "O1_07_compatibility_detector_v1_metadata.json",
                screening_reference_csv=reference_csv)
    if threshold is not None:
        gate.threshold = float(threshold)
    return gate


def validate_rho(req):
    """Step 1a. Reject what can't be scored; warn about small rounding."""
    P = req.P
    if req.item.lower() in SIX_OPTION_ITEMS:
        return RhoCheck("reject", ("six-option chs item: no model exists for it",))
    if P.shape != (4, 4):
        return RhoCheck("reject", (f"rho must be 4x4, got {P.shape}",))
    if not np.isfinite(P).all():
        return RhoCheck("reject", ("rho contains NaN or infinite values",))
    if (P < 0).any():
        return RhoCheck("reject", ("rho contains negative probabilities",))

    error = float(np.max(np.abs(P.sum(axis=1) - 1)))
    shape = calculate_P_structure_descriptors(P)
    certainty, asymmetry = shape["certainty_P"], shape["rya_asymmetry_P"]
    if error > ROW_SUM_LIMIT:
        return RhoCheck("reject", (f"rho row sums off by up to {error:.3f} (limit 0.02)",),
                        error, certainty, asymmetry)

    warnings = []
    if error > 1e-6:
        warnings.append(f"rho row sums rounded (max error {error:.3f}); kept as given")
    if np.any(np.diag(P) < P.max(axis=1)):
        warnings.append("a latent type is not most likely to give its own answer")
    status = "warn" if warnings else "pass"
    if req.rho_source == "json" and req.item in ("sun1", "ry17"):
        warnings.append("JSON and P.RData differ in row 2 col 1 for this item")
    return RhoCheck(status, tuple(warnings), error, certainty, asymmetry)


def screen(req, rho_check, boundary, gate, small_group_n=SMALL_GROUP_N, small_group_ratio=SMALL_GROUP_RATIO):
    """Step 1b. PASS to the network, FLAGGED to the fallback, or REJECT.

    Set small_group_n=None to turn the small-group rule off.
    """
    if rho_check.status == "reject":
        return ScreeningResult(REJECT, rho_check.reasons)

    observed = gate.screen_case(req.x, req.y, req.P, question=req.item)
    if observed["route"] == "invalid_input":
        return ScreeningResult(REJECT, (observed["reason"],))

    reasons = []
    p_observed = observed["classifier"]["incompatibility_probability"]
    if observed["reason"] == "original_pair_degenerate":
        reasons.append("observed group answers all in one category")
    if observed["classifier"]["flagged"]:
        reasons.append(f"classifier flags observed pair (p={p_observed:.2f})")

    p_boundary = float("nan")
    if boundary is not None and np.isfinite(boundary.x).all() and boundary.x.sum() > 0:
        result = gate.screen_case(req.x, req.y, req.P, question=req.item, nn_x=boundary.x, nn_y=boundary.y)
        if result["route"] != "invalid_input":
            p_boundary = result["classifier"]["incompatibility_probability"]
            if result["classifier"]["flagged"]:
                reasons.append(f"classifier flags boundary pair (p={p_boundary:.2f})")

    n, m = req.x.sum(), req.y.sum()
    low, high = TRAINING_SIZES
    outside = not (low <= n <= high and low <= m <= high)
    if outside:
        reasons.append(f"group sizes {int(n)}/{int(m)} outside training range {low}-{high}")

    small, big = min(n, m), max(n, m)
    if small_group_n is not None and small < small_group_n and big / small >= small_group_ratio:
        reasons.append(f"small group ({int(small)}) against a group {big / small:.0f}x larger")

    route = FLAGGED if reasons else PASS
    return ScreeningResult(route, tuple(reasons) or ("screening passed",), p_observed, p_boundary,
                           gate.threshold, outside, {"o1_observed": observed})
