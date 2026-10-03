"""Where the time and memory go when EXP002 scores items (Murat's workstream, step 1).

Before shrinking the model we need to know what is actually slow. One ``predict`` call has
four stages:

    1. features            answer counts -> the 40 inputs (features.py, built with pandas)
    2. normalise           squeeze each input into 0-1 with the training min/max
    3. network             run EXP002 (ONNX)
    4. reverse normalise   turn the two outputs back into shape and rate

This module times each stage on realistic random items, lists the slowest Python functions,
breaks the network down layer by layer (parameters, multiply-adds, memory), and compares
running the network one row at a time (how RYA's ONNX file works today: it only accepts a
single row of 40) with running it on a batch of rows. Finally it measures each speed-up end
to end (``technique_savings``), checks the answers are unchanged, and projects to 200,000
predictions, the scale the toolkit has to stay affordable at.

Nothing here changes the model or its answers. Every faster version is checked against
today's output, and the report says whether it is identical.

    python -m scrp_toolkit.cli profile                 # report on screen + profile_report.json
    python -m scrp_toolkit.cli profile --n-cases 500   # quicker
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import time
import tracemalloc
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnx import numpy_helper

from .config import ProjectPaths
from .features import extract_features
from .inference import UnreliabilityPredictor, batched_onnx_session
from .model import Net

STAGES = ["features", "normalise", "network", "reverse_normalise"]
BATCH_SIZES = (1, 16, 256, 4096)


# ---------------------------------------------------------------------------------------------
# The network outside the ONNX runtime
# ---------------------------------------------------------------------------------------------

def net_from_onnx(onnx_path: str, metadata: dict) -> Net:
    """RYA's ``Net`` with EXP002's trained weights copied in from the ONNX file.

    Same idea as walkthrough Step 7. Handles both the MatMul + Add layers in RYA's file and
    the Gemm layers newer PyTorch versions write.
    """
    graph = onnx.load(onnx_path).graph
    weights = {w.name: numpy_helper.to_array(w) for w in graph.initializer}
    net = Net(n_features=len(metadata["feature_labels"]), n_layers=metadata["n_layers"],
              width=metadata["width"]).eval()
    linears = [m for m in net.linear_relu_stack if isinstance(m, torch.nn.Linear)]

    W, B = [], []
    for node in graph.node:
        params = [weights[i] for i in node.input if i in weights]
        if node.op_type == "MatMul":
            W.append(params[0].T)                        # stored as (in, out)
        elif node.op_type == "Add" and params:
            B.append(params[0])
        elif node.op_type == "Gemm":
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            W.append(params[0] if trans_b else params[0].T)
            B.append(params[1])
    if len(W) != len(linears) or len(B) != len(linears):
        raise ValueError(f"Expected {len(linears)} layers in {onnx_path}, found {len(W)} weights / {len(B)} biases.")
    for lin, w, b in zip(linears, W, B):
        lin.weight.data = torch.tensor(np.ascontiguousarray(w))
        lin.bias.data = torch.tensor(b.copy())
    return net


# ---------------------------------------------------------------------------------------------
# Inputs to profile on
# ---------------------------------------------------------------------------------------------

def sample_cases(p_matrices: Dict[str, np.ndarray], n: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """``n`` random (before counts, after counts, item) triples on the 4-option items.

    Group sizes are spread between a small class (10) and a large school (2,000), and the
    answer shares are random, so the timings aren't tuned to one easy input.
    """
    rng = np.random.default_rng(seed)
    items = [q for q, P in p_matrices.items() if np.shape(P) == (4, 4)]
    cases = []
    for _ in range(n):
        q = items[rng.integers(len(items))]
        n_x, n_y = np.exp(rng.uniform(np.log(10), np.log(2000), size=2)).astype(int)
        x = rng.multinomial(n_x, rng.dirichlet(np.ones(4)))
        y = rng.multinomial(n_y, rng.dirichlet(np.ones(4)))
        cases.append((x, y, q))
    return cases


def _scaled_vector(predictor: UnreliabilityPredictor, x, y, q) -> np.ndarray:
    labels = predictor.metadata["feature_labels"]
    feats = extract_features(x, y, q, predictor.p_matrices).reindex(labels)
    return np.array([predictor._normalise(feats[c], c) for c in labels], dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------------------------

def time_stages(predictor: UnreliabilityPredictor, cases: Sequence) -> Dict[str, float]:
    """Average microseconds per prediction for each of the four stages, plus the whole call.

    The stages repeat ``UnreliabilityPredictor.predict`` step by step; ``predict_total`` times
    the real method on the same inputs, so the two can be compared.
    """
    labels = predictor.metadata["feature_labels"]
    totals = dict.fromkeys(STAGES, 0.0)
    for x, y, q in cases[: min(50, len(cases))]:          # warm-up
        predictor.predict(x, y, q)

    clock = time.perf_counter
    for x, y, q in cases:
        t0 = clock()
        feats = extract_features(x, y, q, predictor.p_matrices).reindex(labels)
        t1 = clock()
        vec = np.array([predictor._normalise(feats[c], c) for c in labels], dtype=np.float32)
        t2 = clock()
        raw = predictor._run(vec)
        t3 = clock()
        predictor._reverse_normalise(raw[0], "shape")
        predictor._reverse_normalise(raw[1], "rate")
        t4 = clock()
        totals["features"] += t1 - t0
        totals["normalise"] += t2 - t1
        totals["network"] += t3 - t2
        totals["reverse_normalise"] += t4 - t3

    t0 = clock()
    for x, y, q in cases:
        predictor.predict(x, y, q)
    total = clock() - t0

    n = len(cases)
    out = {k: v / n * 1e6 for k, v in totals.items()}
    out["predict_total"] = total / n * 1e6
    return out


def hot_functions(predictor: UnreliabilityPredictor, cases: Sequence, top: int = 15) -> pd.DataFrame:
    """The functions ``predict`` spends the most time in (cProfile, own time + time below)."""
    prof = cProfile.Profile()
    prof.enable()
    for x, y, q in cases:
        predictor.predict(x, y, q)
    prof.disable()

    stats = pstats.Stats(prof, stream=io.StringIO())
    n = len(cases)
    rows = []
    for (filename, line, func), (_cc, ncalls, tottime, cumtime, _callers) in stats.stats.items():
        where = f"{os.path.basename(filename)}:{line}" if line else filename
        rows.append({"function": f"{func} ({where})", "calls_per_predict": ncalls / n,
                     "own_us_per_predict": tottime / n * 1e6, "cum_us_per_predict": cumtime / n * 1e6})
    return pd.DataFrame(rows).sort_values("own_us_per_predict", ascending=False).head(top).reset_index(drop=True)


def layer_table(net: Net) -> pd.DataFrame:
    """One row per layer: its size, how many numbers it stores, and the work per input row."""
    rows = []
    for name, mod in net.linear_relu_stack.named_children():
        if not isinstance(mod, torch.nn.Linear):
            continue
        params = mod.weight.numel() + mod.bias.numel()
        rows.append({"layer": name, "in": mod.in_features, "out": mod.out_features, "params": params,
                     "weight_kb_fp32": params * 4 / 1024, "mult_adds_per_row": mod.in_features * mod.out_features})
    df = pd.DataFrame(rows)
    df["share_of_params"] = df["params"] / df["params"].sum()
    return df


def time_network(predictor: UnreliabilityPredictor, net: Net, n_rows: int = 4096,
                 batch_sizes: Sequence[int] = BATCH_SIZES, seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Microseconds per row for the network alone, three ways, at several batch sizes.

    onnx_one_row   today's path: RYA's ONNX file, called once per row
    onnx_batched   the same ONNX model, given many rows per call
    torch_batched  the same weights in PyTorch (``Net``)

    Also returns the largest difference from today's outputs, to show batching changes nothing.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_rows, len(predictor.metadata["feature_labels"]))).astype(np.float32)
    session = batched_onnx_session(predictor.paths.onnx_path)
    in_name = session.get_inputs()[0].name
    clock = time.perf_counter

    for v in X[:50]:
        predictor._run(v)
    t0 = clock()
    reference = np.stack([predictor._run(v) for v in X])
    rows = [{"method": "onnx_one_row", "batch_size": 1, "us_per_row": (clock() - t0) / n_rows * 1e6}]

    diffs = {"onnx_batched": 0.0, "torch_batched": 0.0}
    for bs in batch_sizes:
        for method in ("onnx_batched", "torch_batched"):
            outs = []
            run = (lambda b: session.run(None, {in_name: b})[0]) if method == "onnx_batched" else \
                  (lambda b: net(torch.from_numpy(b)).numpy())
            with torch.no_grad():
                run(X[:bs])                                   # warm-up
                t0 = clock()
                for i in range(0, n_rows, bs):
                    outs.append(run(X[i:i + bs]))
                elapsed = clock() - t0
            out = np.concatenate(outs)
            diffs[method] = max(diffs[method], float(np.abs(out - reference).max()))
            rows.append({"method": method, "batch_size": bs, "us_per_row": elapsed / n_rows * 1e6})

    df = pd.DataFrame(rows)
    df["rows_per_second"] = 1e6 / df["us_per_row"]
    return df, diffs


def memory_profile(predictor: UnreliabilityPredictor, net: Net, cases: Sequence) -> Dict[str, float]:
    """Where memory goes: the model file, the weights, activations, and Python's own overhead."""
    params = sum(p.numel() for p in net.parameters())
    widths = [m.out_features for m in net.linear_relu_stack if isinstance(m, torch.nn.Linear)]
    out = {
        "onnx_file_mb": os.path.getsize(predictor.paths.onnx_path) / 2 ** 20,
        "params": params,
        "weights_mb_fp32": params * 4 / 2 ** 20,
        "weights_mb_fp16": params * 2 / 2 ** 20,
        "weights_mb_int8": params / 2 ** 20,
        # Running a layer needs its input and output in memory at once: 2 x widest layer.
        "activation_kb_per_row": 2 * max(widths) * 4 / 1024,
    }

    # Python memory allocated while making predictions (pandas Series etc.), per prediction.
    tracemalloc.start()
    for x, y, q in cases:
        predictor.predict(x, y, q)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out["python_peak_kb_during_predicts"] = peak / 1024

    # The whole process, if psutil is installed (it isn't in requirements.txt).
    try:
        import psutil

        proc = psutil.Process()
        before = proc.memory_info().rss
        sessions = [ort.InferenceSession(predictor.paths.onnx_path) for _ in range(5)]
        out["onnx_session_rss_mb_each"] = (proc.memory_info().rss - before) / 5 / 2 ** 20
        out["process_rss_mb"] = proc.memory_info().rss / 2 ** 20
        del sessions
    except ImportError:
        pass
    return out


def technique_savings(predictor: UnreliabilityPredictor, cases: Sequence, n_predictions: int) -> pd.DataFrame:
    """Measured time per prediction as each speed-up is added, projected to ``n_predictions``.

    0  today: ``predict`` one item at a time
    1  + network batched: features still one item at a time, then one ONNX call per batch
    2  + features batched: ``predict_batch`` (both stages on whole columns)

    ``identical`` says whether every output matches today's exactly (NaN == NaN).
    """
    labels = predictor.metadata["feature_labels"]
    x = np.array([c[0] for c in cases])
    y = np.array([c[1] for c in cases])
    qs = [c[2] for c in cases]
    clock = time.perf_counter

    def network_batched_only():
        vecs = np.stack([_scaled_vector(predictor, a, b, q) for a, b, q in cases])
        session = batched_onnx_session(predictor.paths.onnx_path)
        raw = session.run(None, {session.get_inputs()[0].name: vecs})[0].astype(np.float64)
        return (np.array([predictor._reverse_normalise(r[0], "shape") for r in raw]),
                np.array([predictor._reverse_normalise(r[1], "rate") for r in raw]))

    methods = [
        ("0 today: predict() one item at a time",
         lambda: tuple(np.array(v) for v in zip(*[predictor.predict(a, b, q) for a, b, q in cases]))),
        ("1 + network batched", network_batched_only),
        ("2 + features batched (predict_batch)", lambda: predictor.predict_batch(x, y, qs)),
    ]
    predictor.predict_batch(x[:10], y[:10], qs[:10])              # warm-up (builds the session)
    rows, reference = [], None
    for name, run in methods:
        t0 = clock()
        out = np.column_stack(run())
        us = (clock() - t0) / len(cases) * 1e6
        if reference is None:
            reference = out
        same = bool(np.array_equal(out, reference, equal_nan=True))
        rows.append({"technique": name, "us_per_prediction": us, "identical": same})

    df = pd.DataFrame(rows)
    df["speed_up_vs_today"] = df["us_per_prediction"].iloc[0] / df["us_per_prediction"]
    df[f"seconds_for_{n_predictions}"] = df["us_per_prediction"] * n_predictions / 1e6
    return df


# ---------------------------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------------------------

def run_profile(paths: ProjectPaths, n_cases: int = 2000, scale: int = 200_000, seed: int = 0) -> dict:
    """Every measurement above, on EXP002 and ``n_cases`` random items.

    Some random groups all give the same answer, so skew/kurtosis come out NaN (a known
    inherited behaviour, see features.py). Those warnings are silenced here; the timings
    still include those inputs.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return _run_profile(paths, n_cases, scale, seed)


def _run_profile(paths: ProjectPaths, n_cases: int, scale: int, seed: int) -> dict:
    t0 = time.perf_counter()
    predictor = UnreliabilityPredictor(paths)
    load_s = time.perf_counter() - t0

    net = net_from_onnx(paths.onnx_path, predictor.metadata)
    cases = sample_cases(predictor.p_matrices, n_cases, seed)

    stages = time_stages(predictor, cases)
    network, diffs = time_network(predictor, net, seed=seed)
    return {
        "model": predictor.describe(),
        "machine": {"cpu_count": os.cpu_count(), "onnxruntime": ort.__version__, "torch": torch.__version__},
        "n_cases": n_cases,
        "load_seconds": load_s,
        "stages_us": stages,
        "hot_functions": hot_functions(predictor, cases[: min(500, n_cases)]),
        "layers": layer_table(net),
        "network": network,
        "max_abs_diff_vs_onnx": diffs,
        "memory": memory_profile(predictor, net, cases[: min(200, n_cases)]),
        "scale": scale,
        "techniques": technique_savings(predictor, cases, scale),
    }


def print_report(result: dict) -> None:
    """The profile as a readable report."""
    pd_opts = ("display.width", 140, "display.max_colwidth", 70, "display.float_format", "{:,.2f}".format)
    with pd.option_context(*pd_opts):
        print(result["model"])
        print(f"Model load: {result['load_seconds']:.3f} s   ({result['n_cases']:,} random items profiled)\n")

        st = result["stages_us"]
        stage_sum = sum(st[s] for s in STAGES)
        print("Time per prediction, by stage")
        for s in STAGES:
            print(f"  {s:<18} {st[s]:9.1f} us   {st[s] / stage_sum:6.1%}")
        print(f"  {'predict() total':<18} {st['predict_total']:9.1f} us\n")

        print("Slowest functions inside predict() (own time)")
        print(result["hot_functions"].head(10).to_string(index=False), "\n")

        layers = result["layers"]
        print(f"Network: {len(layers)} linear layers, {layers['params'].sum():,} parameters, "
              f"{layers['mult_adds_per_row'].sum():,} multiply-adds per row")
        inner = layers.iloc[1:-1]
        print(f"  first layer {layers.iloc[0]['params']:,}  |  {len(inner)} hidden 256x256 layers "
              f"{inner['params'].sum():,} ({inner['share_of_params'].sum():.1%})  |  output layer "
              f"{layers.iloc[-1]['params']:,}\n")

        print("Network alone: microseconds per row")
        print(result["network"].to_string(index=False))
        d = result["max_abs_diff_vs_onnx"]
        print(f"  largest output difference vs today's ONNX: batched ONNX {d['onnx_batched']:.2e}, "
              f"PyTorch {d['torch_batched']:.2e}\n")

        print("Memory")
        for k, v in result["memory"].items():
            print(f"  {k:<32} {v:,.2f}")
        print()

        print(f"Savings per technique, measured on {result['n_cases']:,} items, projected to {result['scale']:,}")
        print(result["techniques"].to_string(index=False))


def to_json_dict(result: dict) -> dict:
    """``result`` with its tables turned into plain lists, ready for json.dump."""
    return {k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v) for k, v in result.items()}
