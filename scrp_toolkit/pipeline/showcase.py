"""Helpers for the short presentation notebook (pipeline_showcase.ipynb).

Everything technical lives here so the notebook cells stay one or two lines long.
The full, step-by-step version is pipeline_walkthrough.ipynb.
"""
from __future__ import annotations

import re
import time
import warnings

import numpy as np
import pandas as pd

from ..config import default_project_root, resolve_project_paths
from .boundary import MEANINGFUL, find_boundary
from .demo import truth_p
from .nulls import simulated_null
from .run import Pipeline, PipelineConfig, build_pipeline
from .scenarios import build_scenarios

PLAIN_NAMES = {
    "A": "Typical comparison",
    "B": "Class of 8 vs national group of 9,000",
    "C": "Class of 25 vs national group",
    "D": "Everyone in one group gave the same answer",
    "E": "Broken reliability matrix (ρ)",
    "F": "Six-option question",
    "G": "Group bigger than the network was trained on",
    "H": "Question reliability from a small retest study",
    "I": "Same comparison, large retest study",
    "J": "Real RYA training case: 6 vs 5,215",
}


def _answer(decision):
    if decision is None:
        return "no answer"
    return "meaningful" if decision == MEANINGFUL else "not meaningful"


def _plain_reason(reasons):
    """Turn the pipeline's technical reasons into one short sentence."""
    text = "; ".join(reasons)
    rules = [
        (r"rho row sums off", "ρ is invalid, so no answer is given"),
        (r"six-option", "the network doesn't cover six-option questions"),
        (r"small group \((\d+)\) against a group (\d+)x larger",
         lambda m: f"small group ({m[1]}) against one {int(m[2]):,}× larger"),
        (r"all in one category", "one group all gave the same answer"),
        (r"outside training range", "a group is bigger than the network was trained on"),
        (r"decision flips", "answer could change with question reliability"),
        (r"classifier flags", "screening doesn't trust the network here"),
    ]
    for pattern, message in rules:
        match = re.search(pattern, text)
        if match:
            return message(match) if callable(message) else message
    return "all checks passed"


def _p(value):
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def _route(result):
    if result.reliability_flag == "RED":
        return "rejected"
    return "fallback (simulation)" if result.screening and result.screening.route == "FLAGGED" else "network"


class Showcase:
    def __init__(self, root=None, epsilon=0.1):
        warnings.filterwarnings("ignore")
        self.root = root or default_project_root()
        self.pipeline = build_pipeline(self.root)
        self.original = Pipeline(self.pipeline.net.predictor, PipelineConfig(mode="direct"), gate=self.pipeline.gate)
        instance_dir = resolve_project_paths(self.root).instance_dir
        self.cases = {c.name[0]: c.request for c in build_scenarios(self.pipeline.net.predictor.p_matrices,
                                                                     instance_dir, epsilon)}
        self.results = {k: self.pipeline.run(r) for k, r in self.cases.items()}
        self.before = {k: self.original.run(r) for k, r in self.cases.items()}

    # --- the overview table -------------------------------------------------------------
    def all_cases(self):
        rows = []
        for key, req in self.cases.items():
            new, old = self.results[key], self.before[key]
            check = truth_p(req, draws=10000) if new.reliability_flag != "RED" else float("nan")
            rows.append({
                "case": key,
                "what it is": PLAIN_NAMES.get(key, req.label),
                "original says": _answer(old.decision),
                "pipeline says": _answer(new.decision),
                "RYA simulation says": "–" if np.isnan(check) else _answer(MEANINGFUL if check < req.alpha else "no"),
                "answered by": _route(new),
                "flag": new.reliability_flag,
                "why": _plain_reason(new.flag_reasons),
            })
        table = pd.DataFrame(rows).set_index("case")
        colours = {"GREEN": "#DCFCE7", "AMBER": "#FEF3C7", "RED": "#FEE2E2"}
        return table.style.apply(lambda col: [f"background-color: {colours.get(v, '')}" for v in col],
                                 subset=["flag"])

    # --- one case in plain words --------------------------------------------------------
    def spotlight(self, key):
        req, new, old = self.cases[key], self.results[key], self.before[key]
        print(f"Case {key}: {PLAIN_NAMES.get(key, req.label)}")
        print(f"  groups        : {int(req.x.sum()):,} vs {int(req.y.sum()):,} students")
        print(f"  original      : {_answer(old.decision)}"
              + ("" if old.p_value is None else f"  ({_p(old.p_value)})"))
        if new.reliability_flag == "RED":
            print(f"  pipeline      : rejected before any calculation")
        else:
            low, high = new.uncertainty_range
            print(f"  pipeline      : {_answer(new.decision)}  ({_p(new.p_value)}, range {'below 0.001' if high < 0.001 else f'{low:.3f}–{high:.3f}'})")
            print(f"  answered by   : {_route(new)}")
        print(f"  flag          : {new.reliability_flag} – {_plain_reason(new.flag_reasons)}")

    def plot_case_b(self):
        import matplotlib.pyplot as plt
        from scipy import stats
        req = self.cases["B"]
        boundary = find_boundary(req)
        network = self.pipeline.net(req, boundary)
        simulation = simulated_null(req, boundary)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.hist(simulation.draws, bins=120, density=True, color="#EA580C", alpha=0.35,
                label="what RYA's own model says can happen by chance")
        xs = np.linspace(0, np.quantile(simulation.draws, 0.995), 400)
        ax.plot(xs, stats.gamma.pdf(xs, network.params["shape"], scale=1 / network.params["rate"]),
                color="#7C3AED", lw=2, label="the network's estimate")
        ax.axvline(boundary.d_norm_observed, color="black", ls="--", label="gap we saw")
        ax.set_xlabel("gap between the two groups"); ax.set_yticks([])
        ax.legend(frameon=False, fontsize=9); ax.set_title("Case B: class of 8 vs national")
        plt.show()

    # --- O3: same answers, different question reliability --------------------------------
    def reliability(self):
        rows = []
        for key in ("H", "I"):
            r = self.results[key]
            low, high = r.uncertainty_range
            rows.append({"case": key, "what it is": PLAIN_NAMES[key], "p-value": round(r.p_value, 3),
                         "p-value range": f"{low:.3f} – {high:.3f}",
                         "crosses 0.05?": "yes" if low < 0.05 <= high else "no",
                         "flag": r.reliability_flag})
        return pd.DataFrame(rows).set_index("case")

    # --- O4: speed ----------------------------------------------------------------------
    def speed(self, repeats=3):
        def per_case(pipe):
            times = []
            for req in self.cases.values():
                start = time.perf_counter()
                for _ in range(repeats):
                    pipe.run(req)
                times.append(1000 * (time.perf_counter() - start) / repeats)
            return np.median(times)

        half = build_pipeline(self.root, weights="fp16")
        same = all(half.run(r).decision == self.results[k].decision for k, r in self.cases.items())
        return pd.DataFrame({
            "": ["original (network only)", "pipeline (all checks)"],
            "typical time per comparison": [f"{per_case(self.original):.1f} ms", f"{per_case(self.pipeline):.0f} ms"],
        }).set_index(""), f"Half-size network (for the browser) gives the same decisions on all cases: {same}"


def start(root=None, epsilon=0.1):
    """Load the pipeline and run all demo cases. Takes about 20 seconds."""
    return Showcase(root, epsilon)
