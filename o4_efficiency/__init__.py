"""O4 (Murat): efficiency and cost optimisation of EXP002.

    batched.py         EfficientPredictor: batched features + network, optional fp16 weights
    profiling.py       where the time and memory go, and what each speed-up saves
    school_scoring.py  score a whole RY25 school in one batched call
    pipeline_cost.py   what the integrated pipeline costs, stage by stage, vs the original path
    FINDINGS.md        what was measured, changed, and tried and dropped
    cli.py             python -m o4_efficiency.cli profile | export-model | pipeline-cost
"""
import integrated_pipeline  # noqa: F401  (pulls RYA's baseline from GitHub on first use)

from .batched import EfficientPredictor, batched_onnx_session, extract_features_batch, onnx_model_bytes
