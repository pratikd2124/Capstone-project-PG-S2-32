"""Stage 1: construct realistic historical subgroup comparisons.

Default behaviour uses within-Post grade comparisons that were already used in
project work. It never treats the entire Pre file as one group and the entire
Post file as the other group.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_DATASET,
    DEFAULT_GROUP_COLUMN,
    DEFAULT_GROUP_PAIRS,
    MIN_GROUP_SIZE,
)
from common import (
    RESULTS_DIR,
    comparable_group_value,
    ensure_output_dirs,
    load_context,
    parse_group_pair,
)

EXPECTED = {1.0, 2.0, 3.0, 4.0}


def _format_group(value) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def build_comparisons(
    assets_root: str,
    *,
    dataset: str = DEFAULT_DATASET,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_pairs: list[tuple[str | int | float, str | int | float]] | None = None,
    min_group_size: int = MIN_GROUP_SIZE,
) -> tuple[pd.DataFrame, list[str]]:
    ctx = load_context(assets_root, require_model=False)
    data_path = ctx.assets.post_csv if dataset.lower() == "post" else ctx.assets.pre_csv
    df = pd.read_csv(data_path, low_memory=False)
    if group_column not in df.columns:
        raise KeyError(f"Grouping column {group_column!r} is not present in {data_path.name}")

    pairs = group_pairs or DEFAULT_GROUP_PAIRS
    rows: list[dict] = []
    skipped: list[str] = []

    # Only items with a valid, normalised 4x4 rho matrix can enter this pipeline.
    candidates = sorted(set(ctx.rho_normalised).intersection(df.columns))
    for item in candidates:
        values = pd.to_numeric(df[item], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            skipped.append(f"{item}: no numeric responses")
            continue
        observed_categories = set(finite.unique().astype(float).tolist())
        if not observed_categories.issubset(EXPECTED):
            skipped.append(
                f"{item}: responses are not restricted to four categories 1-4 "
                f"(observed {sorted(observed_categories)[:10]})"
            )
            continue

        for pair_index, (group_a, group_b) in enumerate(pairs, start=1):
            mask_a = comparable_group_value(df[group_column], group_a)
            mask_b = comparable_group_value(df[group_column], group_b)
            a = values[mask_a].dropna()
            b = values[mask_b].dropna()
            a = a[a.isin([1, 2, 3, 4])]
            b = b[b.isin([1, 2, 3, 4])]
            n_a, n_b = int(len(a)), int(len(b))
            if n_a < min_group_size or n_b < min_group_size:
                skipped.append(
                    f"{item} {_format_group(group_a)} vs {_format_group(group_b)}: "
                    f"group sizes {n_a}/{n_b} below minimum {min_group_size}"
                )
                continue
            counts_a = a.value_counts().reindex([1, 2, 3, 4], fill_value=0).astype(int)
            counts_b = b.value_counts().reindex([1, 2, 3, 4], fill_value=0).astype(int)
            ga, gb = _format_group(group_a), _format_group(group_b)
            comparison_id = f"{dataset.lower()}_{group_column}_{ga}_vs_{gb}_{item}"
            row = {
                "comparison_id": comparison_id,
                "item": item,
                "dataset": data_path.name,
                "group_column": group_column,
                "group_a": ga,
                "group_b": gb,
                "n_a": n_a,
                "n_b": n_b,
                "sample_size_ratio": max(n_a, n_b) / min(n_a, n_b),
            }
            row.update({f"counts_a_{i}": int(counts_a.loc[i]) for i in range(1, 5)})
            row.update({f"counts_b_{i}": int(counts_b.loc[i]) for i in range(1, 5)})
            rows.append(row)

    columns = [
        "comparison_id", "item", "dataset", "group_column", "group_a", "group_b",
        "n_a", "n_b", "sample_size_ratio",
        *[f"counts_a_{i}" for i in range(1, 5)],
        *[f"counts_b_{i}" for i in range(1, 5)],
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        raise RuntimeError("No valid historical subgroup comparisons were created")
    return result, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Create historical GemFinder subgroup comparisons.")
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--dataset", choices=["pre", "post"], default=DEFAULT_DATASET)
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument(
        "--group-pairs", nargs="*", default=None,
        help="Optional A:B subgroup pairs, e.g. 4:7 11:6 12:5",
    )
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    args = parser.parse_args()

    if args.min_group_size < 1:
        raise ValueError("--min-group-size must be positive")
    pairs = None
    if args.group_pairs:
        pairs = [parse_group_pair(x) for x in args.group_pairs]

    ensure_output_dirs()
    result, skipped = build_comparisons(
        args.assets_root,
        dataset=args.dataset,
        group_column=args.group_column,
        group_pairs=pairs,
        min_group_size=args.min_group_size,
    )
    out = RESULTS_DIR / "comparisons.csv"
    result.to_csv(out, index=False)

    notes = RESULTS_DIR / "preparation_notes.md"
    with notes.open("w", encoding="utf-8") as f:
        f.write("# Historical comparison preparation\n\n")
        f.write(f"- Dataset: `{result['dataset'].iloc[0]}`\n")
        f.write(f"- Grouping column: `{args.group_column}`\n")
        f.write(f"- Comparisons created: **{len(result)}**\n")
        f.write(f"- Unique four-option items: **{result['item'].nunique()}**\n")
        f.write("- Whole-population Pre-vs-Post aggregation was not used.\n\n")
        if skipped:
            f.write("## Skipped candidates\n\n")
            for reason in skipped[:500]:
                f.write(f"- {reason}\n")
            if len(skipped) > 500:
                f.write(f"- ... {len(skipped) - 500} additional skip messages omitted.\n")

    print(f"Created {len(result)} comparisons across {result['item'].nunique()} items")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
