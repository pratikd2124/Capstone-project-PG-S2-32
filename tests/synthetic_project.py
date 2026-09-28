"""Build a small fake project folder with the same file layout as the real zip.

Used for end-to-end tests of every CLI command without the real (private) data. The numbers
it produces mean nothing; only the plumbing is being tested.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scrp_toolkit.dataset import build_preprocessed_dataset
from scrp_toolkit.edge_cases import _matrix_to_r_string
from scrp_toolkit.features import extract_features
from scrp_toolkit.model import Net

ITEMS = ["ry9", "ry10", "ry16", "sun12", "cop1", "ph4"]


def build(root: Path, seed: int = 0, n_instance_rows: int = 300, n_layers: int = 2, width: int = 16) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    model_dir = root / "Final Network" / "final_network_model"
    inst_dir = root / "Final Network" / "instance_data"
    for d in (model_dir, inst_dir, root / "Unreliability Matrices", root / "RY25"):
        d.mkdir(parents=True, exist_ok=True)

    # P-matrices
    p = {q: (rng.dirichlet(np.ones(4) * 0.5, size=4) * 0.5 + np.eye(4) * 0.5).tolist() for q in ITEMS}
    p["chs1"] = (np.eye(6) * 0.5 + 0.5 / 6).tolist()  # a 6-option item, like the real chs1-chs6
    (root / "Unreliability Matrices" / "unreliability-matrices.json").write_text(json.dumps(p))

    # instance_data in the real schema
    rows = []
    for i in range(n_instance_rows):
        q = ITEMS[i % len(ITEMS)]
        P = np.array(p[q])
        feats = extract_features(rng.integers(1, 200, 4), rng.integers(1, 200, 4), q, {q: P})
        row = feats.drop([c for c in feats.index if c.startswith("P_")] + ["asymm"]).to_dict()
        row.update({"exp_code": f"orthog_B_1_{i}", "P_matrix": _matrix_to_r_string(P), "P_matrix_question": q,
                    "shape": rng.uniform(1, 10), "rate": rng.uniform(5, 100)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(inst_dir / "orthog_B_1_feature_output_df.csv", index=False)

    # metadata + ONNX model (rank-1 input, like the real EXP002)
    _df, feature_cols, min_max_df = build_preprocessed_dataset(str(inst_dir))
    meta = {"n_layers": n_layers, "width": width, "feature_labels": feature_cols,
            "P_transform": "all_P_asymm", "min_max_df": min_max_df.to_json()}
    (model_dir / "EXP002.json").write_text(json.dumps(meta))

    torch.manual_seed(seed)
    net = Net(len(feature_cols), n_layers, width).eval()
    with torch.no_grad():  # keep outputs inside (0, 1) so reverse-normalised shape/rate stay positive
        net.linear_relu_stack.linear_out.bias.fill_(0.5)
        net.linear_relu_stack.linear_out.weight.mul_(0.01)
    torch.onnx.export(net, torch.rand(len(feature_cols)), str(model_dir / "EXP002_nn_optimal_epoch.onnx"),
                      input_names=["input"], output_names=["output"], dynamo=False)

    # RY25 Pre/Post survey responses
    for name, n in (("RY25_PreMay10.csv", 400), ("RY25_PostMay10.csv", 120)):
        df = pd.DataFrame({"school": rng.choice(["schoolA", "schoolB"], size=n, p=[0.7, 0.3])})
        for q in ITEMS[:-1]:  # leave one item out, like the real 46/59 overlap
            df[q] = rng.integers(1, 5, size=n)
        df["chs1"] = rng.integers(1, 7, size=n)  # must be skipped by scoring, not mis-scored
        df.to_csv(root / "RY25" / name, index=False)
    return root
