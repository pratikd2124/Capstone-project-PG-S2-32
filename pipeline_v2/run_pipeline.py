"""One-command runner for GemFinder pipeline V2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import OFFICIAL_EPSILON, POSTERIOR_DRAWS

HERE = Path(__file__).resolve().parent

STAGES = [
    ("01_prepare_comparisons.py", "Prepare historical comparisons"),
    ("02_screen_inputs.py", "Validate and screen inputs"),
    ("03_standard_exp002.py", "Run EXP002"),
    ("04_fallback_inference.py", "Run fallback where available"),
    ("05_uncertainty_analysis.py", "Run sensitivity analysis"),
    ("06_bayesian_evaluation.py", "Run Bayesian evaluation"),
    ("07_historical_benchmark.py", "Benchmark and time methods"),
    ("08_compare_and_report.py", "Generate figures and final report"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean GemFinder statistical pipeline V2.")
    parser.add_argument("--assets-root", required=True, help="Folder containing extracted assets or the supplied zip archives.")
    parser.add_argument("--draws", type=int, default=POSTERIOR_DRAWS, help="Bayesian posterior draws for Stage 6.")
    parser.add_argument("--official-epsilon", type=float, default=OFFICIAL_EPSILON)
    parser.add_argument("--transition-counts", default=None, help="Optional genuine test-retest transition-count CSV/JSON.")
    args = parser.parse_args()

    assets_root = str(Path(args.assets_root).expanduser().resolve())
    for index, (script, label) in enumerate(STAGES, start=1):
        print(f"\n[{index}/8] {label}", flush=True)
        cmd = [sys.executable, str(HERE / script), "--assets-root", assets_root]
        if script == "06_bayesian_evaluation.py":
            cmd += ["--draws", str(args.draws)]
            if args.official_epsilon is not None:
                cmd += ["--official-epsilon", str(args.official_epsilon)]
            if args.transition_counts:
                cmd += ["--transition-counts", str(Path(args.transition_counts).expanduser().resolve())]
        elif script == "08_compare_and_report.py" and args.official_epsilon is not None:
            cmd += ["--official-epsilon", str(args.official_epsilon)]
        subprocess.run(cmd, cwd=HERE, check=True)

    print("\nPipeline complete.")
    print(f"Results: {HERE / 'results'}")
    print(f"Figures: {HERE / 'figures' / 'final'}")


if __name__ == "__main__":
    main()
