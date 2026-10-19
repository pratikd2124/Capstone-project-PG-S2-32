"""Import Pratik's pipeline_v2 modules without editing them.

His files use top-level imports (``from config import ...``, ``from common import ...``)
because they run as scripts from their own folder. This loader imports them under private
names and puts them in ``sys.modules`` only for the duration of the import, so his files
stay byte-identical to branch ``pratik`` (9f113d6, 15 Oct) and nothing else named
``config`` or ``common`` is affected.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_cache: dict = {}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(f"scrp_o3_{name}", _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def pratik_modules():
    """(config, common, bayes06) from Pratik's pipeline_v2, loaded once."""
    if _cache:
        return _cache["config"], _cache["common"], _cache["bayes06"]
    saved = {k: sys.modules.get(k) for k in ("config", "common")}
    try:
        cfg = _load("config", "config.py")
        sys.modules["config"] = cfg
        common = _load("common", "common.py")
        sys.modules["common"] = common
        bayes06 = _load("bayes06", "06_bayesian_evaluation.py")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    _cache.update(config=cfg, common=common, bayes06=bayes06)
    return cfg, common, bayes06
