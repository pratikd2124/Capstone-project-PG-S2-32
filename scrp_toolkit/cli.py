"""Command-line entry point for the whole pipeline.

Examples
--------
    # Section 1: explore reliability matrices
    python -m scrp_toolkit.cli explore --project-root ./data   # (optional: defaults to <repo>/data) --top-n 10

    # Section 2 + 2b: score a single example, and (if RY25 CSVs are present) a case-study school
    python -m scrp_toolkit.cli score --project-root ./data --out ry25_case_study_scores.csv

    # Section 3: train a fresh network from instance_data/
    python -m scrp_toolkit.cli train --project-root ./data --quick-demo

    # Section 4: compare a freshly-trained network against the shipped EXP002 model
    python -m scrp_toolkit.cli compare --project-root ./data --quick-demo

    # Regression-check the ONNX model's predictions against known_answers/*.json
    python -m scrp_toolkit.cli validate --project-root ./data --known-answers known_answers/ry9_example.json
    python -m scrp_toolkit.cli validate --project-root ./data --known-answers known_answers/ry9_example.json --capture

    # RYA baseline: verify files (SHA-1), preprocessing (EXP002 scaling) and every output
    python -m scrp_toolkit.cli baseline --project-root ./data
    python -m scrp_toolkit.cli baseline --project-root ./data --capture scrp_toolkit/known_answers/rya_baseline.json
    python -m scrp_toolkit.cli baseline --project-root ./data --baseline scrp_toolkit/known_answers/murat_notebook_reference.json

    # Generate edge-case data (response-count cases + instance_data-schema rows for a
    # bad-input detector), and check the response-count cases against the real pipeline
    python -m scrp_toolkit.cli gen-edge-cases --project-root ./data --check-response-cases
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import default_project_root, resolve_project_paths
from .reliability import load_p_matrices, score_asymmetry
from .validate import ValidationCase, load_known_answers, print_report, validate


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project-root", default=None,
                   help="Folder containing the customer's data files (default: $SCRP_DATA, else <repo>/data)")
    p.add_argument("--zip", default=None, help="Optional zip to extract into --project-root first")


def cmd_explore(args: argparse.Namespace) -> int:
    paths = resolve_project_paths(args.project_root, args.zip)
    p_matrices = load_p_matrices(paths.pmatrices_path)
    scores = score_asymmetry(p_matrices)
    print(f"Loaded reliability matrices for {len(p_matrices)} survey items.")
    print(f"\nTop {args.top_n} noisiest items (highest asymm):")
    print(scores.head(args.top_n).to_string())
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .inference import UnreliabilityPredictor
    from .scoring import score_school

    paths = resolve_project_paths(args.project_root, args.zip)
    predictor = UnreliabilityPredictor(paths)
    print(predictor.describe())

    x_example = np.array([40, 52, 7, 14])
    y_example = np.array([9, 17, 54, 12])
    shape_hat, rate_hat = predictor.predict(x_example, y_example, "ry9")
    print(f"\nSingle example (item ry9): shape={shape_hat:.4f}  rate={rate_hat:.4f}  gamma_mean={shape_hat / rate_hat:.4f}")

    if paths.have_ry25_data:
        scores = score_school(predictor, paths)
        print(f"\nScored {len(scores)} items for school '{scores.attrs.get('school')}'.")
        print(scores.head(10).to_string())
        if args.out:
            scores.to_csv(args.out, index=False)
            print(f"\nSaved: {args.out}")
    else:
        print("\nRY25 CSVs not found under --project-root; skipping bulk scoring.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train_from_instance_data

    paths = resolve_project_paths(args.project_root, args.zip)
    config = TrainConfig(num_epochs=15 if args.quick_demo else 1000)
    result = train_from_instance_data(paths.instance_dir, config)
    print(f"\nFinal train_loss={result['train_loss_history'][-1]:.4f}  "
          f"test_loss={result['test_loss_history'][-1]:.4f}")
    if args.save_model:
        import torch

        torch.save(result["model"].state_dict(), args.save_model)
        print(f"Saved model weights: {args.save_model}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_against_exp002
    from .inference import UnreliabilityPredictor
    from .train import TrainConfig, train_from_instance_data

    paths = resolve_project_paths(args.project_root, args.zip)
    config = TrainConfig(num_epochs=15 if args.quick_demo else 1000)
    trained = train_from_instance_data(paths.instance_dir, config)
    predictor = UnreliabilityPredictor(paths)

    result = compare_against_exp002(
        demo_model=trained["model"],
        predictor=predictor,
        test_df=trained["test_df"],
        feature_columns=trained["feature_columns"],
        min_max_train_df=trained["min_max_train_df"],
        device=trained["device"],
    )
    print(f"Mean absolute error (shape):")
    print(f"  Model trained just now ({config.num_epochs} epochs) : {result['mae_demo_shape']:.3f}")
    print(f"  Pre-trained EXP002 model                    : {result['mae_exp002_shape']:.3f}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .inference import UnreliabilityPredictor

    paths = resolve_project_paths(args.project_root, args.zip)
    predictor = UnreliabilityPredictor(paths)
    cases = load_known_answers(args.known_answers)

    if args.capture:
        # Run the model now and overwrite the file with the current output as the new baseline.
        captured = []
        for case in cases:
            shape_hat, rate_hat = predictor.predict(np.array(case.x), np.array(case.y), case.question)
            captured.append({
                "question": case.question, "x": case.x, "y": case.y,
                "expected_shape": shape_hat, "expected_rate": rate_hat, "label": case.label,
            })
        Path(args.known_answers).write_text(json.dumps(captured, indent=2))
        print(f"Captured {len(captured)} case(s) as the new baseline in {args.known_answers}.")
        return 0

    results = validate(predictor, cases, rel_tol=args.rel_tol)
    all_passed = print_report(results)
    return 0 if all_passed else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    from .baseline import (RYA_BASELINE, capture_baseline, check_against_baseline,
                           load_baseline, print_summary, report_frame)

    paths = resolve_project_paths(args.project_root, args.zip)
    baseline_path = Path(args.baseline) if args.baseline else RYA_BASELINE
    baseline = load_baseline(baseline_path)

    if args.capture:
        try:
            captured = capture_baseline(paths, baseline)
        except (RuntimeError, FileNotFoundError) as exc:
            print(exc)
            return 1
        Path(args.capture).write_text(json.dumps(captured, indent=2))
        print(f"Captured the RYA baseline: {args.capture}")
        return 0

    print(f"Checking against: {baseline_path.name}")
    try:
        checks = check_against_baseline(paths, baseline, include_dataset=not args.skip_dataset)
    except FileNotFoundError as exc:
        print(f"Cannot run the baseline: {exc}")
        return 1
    ok = print_summary(checks)
    if args.report:
        report_frame(checks).to_csv(args.report, index=False)
        print(f"Full report: {args.report}")
    return 0 if ok else 1


def cmd_gen_edge_cases(args: argparse.Namespace) -> int:
    from . import edge_cases

    gen_argv = ["--out-dir", args.out_dir, "--n-per-case", str(args.n_per_case), "--seed", str(args.seed)]
    if args.project_root:
        gen_argv += ["--project-root", args.project_root]
    if args.check_response_cases:
        gen_argv += ["--check-response-cases"]
    return edge_cases.main(gen_argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scrp_toolkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("explore", help="Section 1: reliability-matrix asymmetry ranking")
    _add_common_args(p)
    p.add_argument("--top-n", type=int, default=10)
    p.set_defaults(func=cmd_explore)

    p = sub.add_parser("score", help="Section 2/2b: single example + case-study school scoring")
    _add_common_args(p)
    p.add_argument("--out", default=None, help="CSV path to save the case-study school's scores")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("train", help="Section 3: train a fresh Net from instance_data/")
    _add_common_args(p)
    p.add_argument("--quick-demo", action="store_true", help="15 epochs instead of the original 1000")
    p.add_argument("--save-model", default=None, help="Path to save the trained state_dict")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("compare", help="Section 4: freshly-trained Net vs shipped EXP002 ONNX model")
    _add_common_args(p)
    p.add_argument("--quick-demo", action="store_true", help="15 epochs instead of the original 1000")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("validate", help="Regression-check ONNX predictions against known_answers/*.json")
    _add_common_args(p)
    p.add_argument("--known-answers", required=True)
    p.add_argument("--rel-tol", type=float, default=0.01)
    p.add_argument("--capture", action="store_true", help="Overwrite --known-answers with today's output")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("baseline", help="RYA baseline: file SHA-1s, EXP002 scaling, and every output; PASS/FAIL per check")
    _add_common_args(p)
    p.add_argument("--baseline", default=None, help="Baseline JSON (default: known_answers/rya_baseline.json)")
    p.add_argument("--report", default="baseline_report.csv", help="CSV with one row per check")
    p.add_argument("--skip-dataset", action="store_true", help="Skip the scaling + dataset sections (the slow instance_data rebuild)")
    p.add_argument("--capture", default=None, metavar="OUT_JSON", help="Record a full baseline (only if files + scaling pass) to OUT_JSON")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("gen-edge-cases", help="Generate edge-case data (response-count cases + instance_data-schema rows)")
    p.add_argument("--project-root", default=None, help="Needed only with --check-response-cases")
    p.add_argument("--zip", default=None)
    p.add_argument("--out-dir", default="edge_cases")
    p.add_argument("--n-per-case", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-response-cases", action="store_true")
    p.set_defaults(func=cmd_gen_edge_cases)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "project_root", "unset") is None and args.command != "gen-edge-cases":
        args.project_root = default_project_root()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
