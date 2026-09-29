"""The RYA baseline: proving our copy behaves exactly like RYA's system.

``python -m scrp_toolkit.cli baseline`` runs everything in here and compares the results with
the numbers recorded in known_answers/rya_baseline.json. Every comparison is one "check" in the
report. It says PASS or FAIL, and the command exits with an error if anything fails.

The checks come in sections:

    files       each of the 9 input files has exactly the fingerprint (SHA-1) Box reports for
                the customer's copy. One changed byte, or a stray extra CSV, fails.
    explore     the matrices file: item count and the 10 most lopsided items
    model       the model's own description (layers, width, inputs)
    ry9         RYA's worked example gives the recorded shape and rate
    case_study  every item for a real RY25 school gives the recorded numbers
    scaling     our data preparation reproduces the 84 min/max values stored with the model
                (EXP002.json), which proves we prepare data exactly as RYA did
    dataset     row counts, and EXP002's error on the fixed train/test split

Recording a new baseline (``--capture``) is only allowed when the files and scaling sections
pass, so a baseline can only ever come from the customer's files and RYA's preparation.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .reliability import load_p_matrices, score_asymmetry

# How close is "the same"? The model works in 32-bit numbers, so allow a relative difference of
# 0.001% for tiny differences between computers and library versions.
RTOL = 1e-5

KNOWN_ANSWERS = Path(__file__).parent / "known_answers"
RYA_BASELINE = KNOWN_ANSWERS / "rya_baseline.json"
MURAT_REFERENCE = KNOWN_ANSWERS / "murat_notebook_reference.json"

# Which ProjectPaths field holds each fingerprinted file.
_PATH_FIELDS = {
    "onnx_path": "EXP002_nn_optimal_epoch.onnx",
    "metadata_path": "EXP002.json",
    "pmatrices_path": "unreliability-matrices.json",
    "pre_csv_path": "RY25_PreMay10.csv",
    "post_csv_path": "RY25_PostMay10.csv",
}


@dataclass
class Check:
    """One line of the report: what we expected, what we got, and whether they match."""

    section: str
    check: str
    expected: object
    got: object
    passed: bool
    note: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _close(got, expected, atol: float, rtol: float = RTOL) -> bool:
    """True if two numbers agree within the tolerance (False if either isn't a number)."""
    try:
        return bool(np.isclose(float(got), float(expected), rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return False


def _num(section, name, got, expected, atol, rtol=RTOL) -> Check:
    """A check that two numbers are close."""
    return Check(section, name, expected, None if got is None else round(float(got), 6), _close(got, expected, atol, rtol))


def _eq(section, name, got, expected, note="") -> Check:
    """A check that two things are exactly equal."""
    return Check(section, name, expected, got, got == expected, note)


def _missing(section, name, expected, why) -> Check:
    """A failed check for something that couldn't be found at all."""
    return Check(section, name, expected, None, False, why)


def sha1_of(path: str | Path) -> str:
    """The file's SHA-1 fingerprint, the same value Box shows for it."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_map(paths: ProjectPaths) -> dict:
    """{file name: where it is on this computer (or None)} for every fingerprinted file."""
    files = {name: getattr(paths, field) for field, name in _PATH_FIELDS.items()}
    if paths.instance_dir:
        for p in sorted(Path(paths.instance_dir).glob("*.csv")):
            files[p.name] = str(p)
    return files


# ---------------------------------------------------------------------------
# One function per section of the report
# ---------------------------------------------------------------------------


def check_files(paths: ProjectPaths, manifest: dict) -> List[Check]:
    """Compare every file's fingerprint with the one Box reports."""
    files = _file_map(paths)
    checks = []
    for name, expected_sha1 in manifest.items():
        local = files.get(name)
        if local is None:
            checks.append(_missing("files", name, expected_sha1, "file not found under --project-root"))
        else:
            checks.append(_eq("files", name, sha1_of(local), expected_sha1))
    extra = sorted(n for n in files if n.endswith(".csv") and n not in manifest and files[n]
                   and Path(files[n]).parent == Path(paths.instance_dir or ""))
    # An extra CSV in the training-data folder would silently change the training data.
    for name in extra:
        checks.append(Check("files", name, "not present", "present", False,
                            "extra CSV in instance_data/ -- remove it (e.g. edge-case rows)"))
    return checks


def run_scaling(paths: ProjectPaths, metadata: dict, normalise_rate: bool = True):
    """Prepare the training data our way, and fetch the min/max RYA stored with the model."""
    from .dataset import build_preprocessed_dataset

    df, feature_cols, rebuilt = build_preprocessed_dataset(paths.instance_dir, normalise_rate=normalise_rate)
    stored = pd.read_json(io.StringIO(metadata["min_max_df"]))
    return df, feature_cols, rebuilt, stored


def check_scaling(df, feature_cols, rebuilt: pd.DataFrame, stored: pd.DataFrame, metadata: dict) -> List[Check]:
    """Our min/max for every column vs the ones stored in EXP002.json (84 numbers), plus the order."""
    checks = [_eq("scaling", "feature_order", list(feature_cols), list(metadata["feature_labels"]))]
    r, s = rebuilt.set_index("column"), stored.set_index("column")
    checks.append(_eq("scaling", "columns", list(r.index), list(s.index)))
    for col in s.index:
        for stat in ("min", "max"):
            got = r[stat].get(col) if col in r.index else None
            # EXP002.json stores these rounded to 10 decimal places.
            checks.append(_num("scaling", f"{col}.{stat}", got, s.loc[col, stat], atol=1e-9, rtol=1e-9))
    return checks


def run_explore(paths: ProjectPaths, top_n: int = 10) -> dict:
    """Item counts and the ``top_n`` most lopsided matrices."""
    p = load_p_matrices(paths.pmatrices_path)
    scores = score_asymmetry(p)
    return {"n_items": int(len(p)),
            "n_four_option_items": int(sum(np.shape(m) == (4, 4) for m in p.values())),
            "top_asymm": {q: float(v) for q, v in scores.head(top_n).items()}}


def run_ry9(predictor, case: dict) -> dict:
    """Run the worked example through the model."""
    shape, rate = predictor.predict(np.array(case["x"]), np.array(case["y"]), case["question"])
    return {"question": case["question"], "x": case["x"], "y": case["y"],
            "shape": shape, "rate": rate, "gamma_mean": shape / rate}


def run_case_study(predictor, paths: ProjectPaths) -> Optional[pd.DataFrame]:
    """Score every item for the RY25 case-study school (None if the survey files aren't there)."""
    from .scoring import score_school

    if not paths.have_ry25_data:
        return None
    return score_school(predictor, paths)


def run_split_error(predictor, df, feature_cols, min_max_df, seed: int, train_frac: float) -> dict:
    """EXP002's average error (shape and rate) on the test rows of a fixed split. No training.

    RYA picked EXP002's split at random and never saved it, so many of these rows were probably
    in EXP002's own training data. The number is a fixed reference for spotting changes, not a
    measure of how well EXP002 does on unseen data.
    """
    from .compare import _rescale, mae
    from .dataset import train_test_split_like_original

    train_df, test_df = train_test_split_like_original(df, train_frac=train_frac, random_seed=seed)
    raw = test_df[feature_cols].copy()
    for c in feature_cols:
        raw[c] = _rescale(min_max_df, c, raw[c].values)
    preds = np.array([predictor.predict_raw_features(row) for _, row in raw.iterrows()])
    true_shape = _rescale(min_max_df, "shape", test_df["shape"].values)
    true_rate = _rescale(min_max_df, "rate", test_df["rate"].values)
    return {"n_rows": int(len(df)), "n_train": int(len(train_df)), "n_test": int(len(test_df)),
            "exp002_shape_mae": mae(preds[:, 0], true_shape), "exp002_rate_mae": mae(preds[:, 1], true_rate)}


# ---------------------------------------------------------------------------
# Running all the checks, or recording a new baseline
# ---------------------------------------------------------------------------


def _load_metadata(paths: ProjectPaths) -> dict:
    """Read EXP002.json."""
    with open(paths.metadata_path) as f:
        return json.load(f)


def check_against_baseline(paths: ProjectPaths, baseline: dict, include_dataset: bool = True) -> List[Check]:
    """Run every section the baseline file has numbers for, and return all the checks.

    include_dataset=False skips the slow part (rebuilding the training data).
    """
    from .inference import UnreliabilityPredictor

    checks: List[Check] = []
    if "files" in baseline:
        checks += check_files(paths, baseline["files"])

    exp = baseline.get("explore")
    if exp:
        got = run_explore(paths, top_n=len(exp["top_asymm"]))
        checks.append(_eq("explore", "n_items", got["n_items"], exp["n_items"]))
        if "n_four_option_items" in exp:
            checks.append(_eq("explore", "n_four_option_items", got["n_four_option_items"], exp["n_four_option_items"]))
        checks.append(_eq("explore", "top_order", list(got["top_asymm"]), list(exp["top_asymm"])))
        for q, v in exp["top_asymm"].items():
            checks.append(_num("explore", f"asymm[{q}]", got["top_asymm"].get(q), v, exp["atol"]))

    # Everything after this point needs the model itself.
    if not (paths.onnx_path and paths.metadata_path):
        checks.append(_missing("model", "EXP002 model files", "present",
                               "EXP002_nn_optimal_epoch.onnx / EXP002.json not found -- later sections skipped"))
        return checks
    metadata = _load_metadata(paths)
    predictor = UnreliabilityPredictor(paths)
    if "model" in baseline:
        checks.append(_eq("model", "describe", predictor.describe(), baseline["model"]["describe"]))

    ry9 = baseline.get("ry9_example")
    if ry9:
        got = run_ry9(predictor, ry9)
        for k in ("shape", "rate", "gamma_mean"):
            if k in ry9:
                checks.append(_num("ry9", k, got[k], ry9[k], ry9["atol"]))

    cs = baseline.get("case_study")
    if cs:
        scores = run_case_study(predictor, paths)
        if scores is None:
            checks.append(_missing("case_study", "school", cs["school"], "RY25 CSVs not found"))
        else:
            checks.append(_eq("case_study", "school", str(scores.attrs.get("school")), cs["school"]))
            if "n_items_scored" in cs:
                checks.append(_eq("case_study", "n_items_scored", int(len(scores)), cs["n_items_scored"]))
            by_q = scores.set_index("question")
            for item in cs["items"]:
                q = item["question"]
                if q not in by_q.index:
                    checks.append(_missing("case_study", q, "scored", "item not scored"))
                    continue
                row = by_q.loc[q]
                for k in ("n_pre", "n_post"):
                    checks.append(_eq("case_study", f"{q}.{k}", int(row[k]), item[k]))
                for k in ("shape_hat", "rate_hat", "gamma_mean", "asymm"):
                    if k in item:
                        checks.append(_num("case_study", f"{q}.{k}", row[k], item[k], cs["atol"]))

    ds = baseline.get("dataset")
    if include_dataset and (ds or baseline.get("check_scaling")):
        if not paths.instance_dir:
            checks.append(_missing("scaling", "instance_data", "present", "instance_data/ not found"))
        else:
            df, feature_cols, rebuilt, stored = run_scaling(paths, metadata)
            if baseline.get("check_scaling"):
                checks += check_scaling(df, feature_cols, rebuilt, stored, metadata)
            if ds:
                got = run_split_error(predictor, df, feature_cols, rebuilt, ds["split_seed"], ds["train_frac"])
                for k in ("n_rows", "n_train", "n_test"):
                    checks.append(_eq("dataset", k, got[k], ds[k]))
                for k in ("exp002_shape_mae", "exp002_rate_mae"):
                    if k in ds:
                        checks.append(_num("dataset", k, got[k], ds[k], ds["mae_atol"]))
    return checks


def report_frame(checks: List[Check]) -> pd.DataFrame:
    """All checks as a table: this is what gets saved as baseline_report.csv."""
    return pd.DataFrame([{"section": c.section, "check": c.check, "expected": c.expected,
                          "got": c.got, "status": c.status, "note": c.note} for c in checks])


def print_summary(checks: List[Check]) -> bool:
    """Print one PASS/FAIL line per section, then any mismatches. True if everything passed."""
    df = report_frame(checks)
    fails = df[df["status"] == "FAIL"]
    for section, grp in df.groupby("section", sort=False):
        n_fail = int((grp["status"] == "FAIL").sum())
        print(f"[{'PASS' if n_fail == 0 else 'FAIL'}] {section:<11} {len(grp) - n_fail}/{len(grp)} checks passed")
    if len(fails):
        print("\nMismatches:")
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(fails[["section", "check", "expected", "got", "note"]].head(60).to_string(index=False))
        if len(fails) > 60:
            print(f"... {len(fails) - 60} more in the CSV report")
    print(f"\nTOTAL: {int((df['status'] == 'PASS').sum())}/{len(df)} checks passed -> "
          f"{'BASELINE MATCHES' if fails.empty else 'BASELINE MISMATCH'}")
    return fails.empty


def capture_baseline(paths: ProjectPaths, template: dict) -> dict:
    """Record a fresh baseline from the current files (only needed if RYA sends new files).

    Refuses (RuntimeError) unless every file matches Box's fingerprints and our data preparation
    reproduces EXP002.json, so a recorded baseline is always genuinely RYA's.
    """
    from .inference import UnreliabilityPredictor

    gate = check_files(paths, template["files"]) if "files" in template else []
    paths.require("onnx_path", "metadata_path", "instance_dir")
    metadata = _load_metadata(paths)
    df, feature_cols, rebuilt, stored = run_scaling(paths, metadata)
    gate += check_scaling(df, feature_cols, rebuilt, stored, metadata)
    # The gate: don't record anything unless the inputs and the preparation are RYA's.
    bad = [c for c in gate if not c.passed]
    if bad:
        print_summary(gate)
        raise RuntimeError(f"Not capturing: {len(bad)} file/scaling check(s) failed -- inputs are not RYA's.")

    predictor = UnreliabilityPredictor(paths)
    out = {k: template[k] for k in ("_source", "files", "check_scaling") if k in template}
    out["_captured"] = "Outputs recorded by `scrp_toolkit.cli baseline --capture` after files + scaling passed."

    exp = run_explore(paths, top_n=len(template["explore"]["top_asymm"]))
    out["explore"] = {"n_items": exp["n_items"], "n_four_option_items": exp["n_four_option_items"],
                      "top_asymm": {q: round(v, 6) for q, v in exp["top_asymm"].items()},
                      "atol": template["explore"]["atol"]}
    out["model"] = {"describe": predictor.describe()}

    t = template.get("ry9_example", {"question": "ry9", "x": [40, 52, 7, 14], "y": [9, 17, 54, 12], "atol": 1e-6})
    ry9 = run_ry9(predictor, t)
    out["ry9_example"] = {**{k: ry9[k] for k in ("question", "x", "y")},
                          **{k: round(ry9[k], 8) for k in ("shape", "rate", "gamma_mean")}, "atol": 1e-6}

    scores = run_case_study(predictor, paths)
    if scores is not None:
        out["case_study"] = {
            "school": str(scores.attrs.get("school")), "n_items_scored": int(len(scores)), "atol": 1e-6,
            "items": [{"question": r.question, "n_pre": int(r.n_pre), "n_post": int(r.n_post),
                       **{k: round(float(getattr(r, k)), 8) for k in ("shape_hat", "rate_hat", "gamma_mean", "asymm")}}
                      for r in scores.itertuples()],  # every scored item, not a sample
        }

    ds_t = template.get("dataset", {"split_seed": 42, "train_frac": 0.8, "mae_atol": 1e-6})
    got = run_split_error(predictor, df, feature_cols, rebuilt, ds_t["split_seed"], ds_t["train_frac"])
    out["dataset"] = {"split_seed": ds_t["split_seed"], "train_frac": ds_t["train_frac"],
                      **{k: got[k] for k in ("n_rows", "n_train", "n_test")},
                      "exp002_shape_mae": round(got["exp002_shape_mae"], 8),
                      "exp002_rate_mae": round(got["exp002_rate_mae"], 8), "mae_atol": 1e-6}
    return out


def load_baseline(path: str | Path) -> dict:
    """Read a baseline file (e.g. known_answers/rya_baseline.json)."""
    with open(path) as f:
        return json.load(f)
