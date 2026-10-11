"""Create controlled scenarios for standalone sensitivity experiments.

No raw student-level CSV is required. Response proportions are held fixed while
sample size is changed, making the sample-size experiment easy to interpret.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from standalone_common import (
    DEFAULT_ITEM,
    DEFAULT_PRE,
    DEFAULT_POST,
    RESULTS_DIR,
    counts_from_proportions,
    ensure_output_dirs,
    proportions_from_counts,
)


def build_sample_size_scenarios(item, pre, post, sizes):
    p_pre = proportions_from_counts(pre)
    p_post = proportions_from_counts(post)
    rows = []
    for n in sizes:
        x = counts_from_proportions(p_pre, n)
        y = counts_from_proportions(p_post, n)
        rows.append({
            "scenario": f"n_{n}", "item": item,
            "n_pre": int(x.sum()), "n_post": int(y.sum()),
            **{f"pre_{i+1}": int(x[i]) for i in range(4)},
            **{f"post_{i+1}": int(y[i]) for i in range(4)},
        })
    return pd.DataFrame(rows)


def build_edge_cases(item, pre, post):
    p_pre, p_post = proportions_from_counts(pre), proportions_from_counts(post)
    cases = {
        "normal": (counts_from_proportions(p_pre, 100), counts_from_proportions(p_post, 100)),
        "zero_samples": (np.zeros(4, dtype=int), np.zeros(4, dtype=int)),
        "one_respondent": (np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])),
        "single_category": (np.array([100, 0, 0, 0]), np.array([100, 0, 0, 0])),
        "missing_category": (np.array([40, 30, 30, 0]), np.array([35, 35, 30, 0])),
        "pre_post_imbalance": (np.array([400, 300, 200, 100]), np.array([4, 3, 2, 1])),
        "very_large_n": (counts_from_proportions(p_pre, 20_000), counts_from_proportions(p_post, 20_000)),
    }
    rows = []
    for name, (x, y) in cases.items():
        rows.append({
            "scenario": name, "item": item,
            "n_pre": int(x.sum()), "n_post": int(y.sum()),
            **{f"pre_{i+1}": int(x[i]) for i in range(4)},
            **{f"post_{i+1}": int(y[i]) for i in range(4)},
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", default=DEFAULT_ITEM)
    parser.add_argument("--pre", nargs=4, type=float, default=DEFAULT_PRE.tolist())
    parser.add_argument("--post", nargs=4, type=float, default=DEFAULT_POST.tolist())
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[20, 50, 100, 250, 500, 1000, 5000, 10_000, 20_000])
    args = parser.parse_args()

    ensure_output_dirs()
    pre, post = np.asarray(args.pre), np.asarray(args.post)
    sample = build_sample_size_scenarios(args.item, pre, post, args.sizes)
    edge = build_edge_cases(args.item, pre, post)
    sample.to_csv(RESULTS_DIR / "sample_size_scenarios.csv", index=False)
    edge.to_csv(RESULTS_DIR / "edge_case_scenarios.csv", index=False)
    print(sample.to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'sample_size_scenarios.csv'}")
    print(f"Wrote {RESULTS_DIR / 'edge_case_scenarios.csv'}")


if __name__ == "__main__":
    main()
