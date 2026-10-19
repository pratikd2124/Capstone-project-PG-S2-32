"""The integrated pipeline: RYA's method with O1 screening, O2 fallback, O3 uncertainty and
O4 efficiency around it. RYA's baseline code is pulled from GitHub on first use (baseline.py).
"""
from .baseline import ensure_baseline

ensure_baseline()
