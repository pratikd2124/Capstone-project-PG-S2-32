"""O4 commands.

    python -m o4_efficiency.cli profile                  # where EXP002's time and memory go
    python -m o4_efficiency.cli export-model             # EXP002 with fp16 weights, for the browser
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrated_pipeline.baseline import data_root
from scrp_toolkit.config import resolve_project_paths


def cmd_profile(args) -> int:
    from .profiling import print_report, run_profile, to_json_dict

    paths = resolve_project_paths(args.project_root or data_root())
    paths.require("onnx_path", "metadata_path")
    result = run_profile(paths, n_cases=args.n_cases, scale=args.scale)
    print_report(result)
    if args.out:
        Path(args.out).write_text(json.dumps(to_json_dict(result), indent=2, default=float))
        print(f"\nSaved: {args.out}")
    return 0


def cmd_export_model(args) -> int:
    from .batched import onnx_model_bytes

    paths = resolve_project_paths(args.project_root or data_root())
    paths.require("onnx_path")
    data = onnx_model_bytes(paths.onnx_path, batched=True, weights=args.weights)
    Path(args.out).write_bytes(data)
    print(f"Saved: {args.out}  ({len(data) / 2**20:.2f} MB, weights {args.weights}, input [batch, 40])")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="Where EXP002's time and memory go")
    p.add_argument("--project-root", default=None, help="Baseline data folder (default: downloaded from GitHub)")
    p.add_argument("--n-cases", type=int, default=2000, help="Random items to time (default 2000)")
    p.add_argument("--scale", type=int, default=200_000, help="Predictions to project the cost to")
    p.add_argument("--out", default="profile_report.json", help="JSON report")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("export-model", help="EXP002 as a batch-ready ONNX file (fp16 weights by default)")
    p.add_argument("--project-root", default=None, help="Baseline data folder (default: downloaded from GitHub)")
    p.add_argument("--weights", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--out", default="exp002_fp16.onnx")
    p.set_defaults(func=cmd_export_model)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
