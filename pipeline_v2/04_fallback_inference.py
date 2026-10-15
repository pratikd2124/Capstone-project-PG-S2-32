"""Stage 4: production fallback routing interface.

No fallback statistical method is invented here. Replace ``run_fallback`` only
when the team's real selected fallback implementation is supplied.
"""

from __future__ import annotations

import argparse
import pandas as pd

from common import RESULTS_DIR, ensure_output_dirs

FALLBACK_STATUS = "NOT INTEGRATED — awaiting team fallback implementation"


def run_fallback(comparison: pd.Series) -> dict:
    """Interface for the team-owned fallback method.

    Return a dictionary with real fallback estimates only after integrating the
    team's validated implementation. Until then, unsupported numeric outputs are
    intentionally absent.
    """
    return {
        "fallback_used": False,
        "fallback_status": FALLBACK_STATUS,
        "fallback_runtime_ms": None,
        "fallback_result": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Route incompatible comparisons to fallback inference.")
    parser.add_argument("--assets-root", required=False, help="Accepted for pipeline interface consistency.")
    args = parser.parse_args()

    ensure_output_dirs()
    comparisons = pd.read_csv(RESULTS_DIR / "comparisons.csv")
    screening = pd.read_csv(RESULTS_DIR / "screening_results.csv")
    df = comparisons.merge(screening[["comparison_id", "screening_status"]], on="comparison_id", how="left")

    rows: list[dict] = []
    for _, row in df.iterrows():
        status = str(row["screening_status"])
        if status == "SAFE":
            rows.append({
                "comparison_id": row["comparison_id"],
                "inference_path": "EXP002",
                "fallback_required": False,
                "fallback_used": False,
                "fallback_status": "NOT_REQUIRED",
                "fallback_runtime_ms": None,
                "fallback_result": None,
            })
        else:
            result = run_fallback(row)
            rows.append({
                "comparison_id": row["comparison_id"],
                "inference_path": "FALLBACK_REQUIRED",
                "fallback_required": True,
                **result,
            })

    out_df = pd.DataFrame(rows)
    out = RESULTS_DIR / "fallback_results.csv"
    out_df.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(FALLBACK_STATUS)


if __name__ == "__main__":
    main()
