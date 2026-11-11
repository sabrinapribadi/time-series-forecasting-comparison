"""CPU latency of the zero-shot models on an idle machine (ADR-008 reporting): 30 test origins each, one forecast per call.

    ~/.venvs/tsfc-fm/bin/python scripts/rolling_latency.py      # writes data/forecasts/ETT/rolling/latency_idle.json
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rolling_foundation import MODELS  # noqa: E402
from src.evaluation import rolling as R  # noqa: E402

N = 30


def main() -> None:
    _, y = R.load_ot()
    orig = R.origins("test")[:N]
    out = {"origins": N, "cpu": platform.processor() or platform.machine(), "torch_threads": torch.get_num_threads(), "models": {}}
    for name in ("chronos2", "timesfm25", "moirai2", "chronos2_cov"):
        m = MODELS[name]()
        m(y, orig[0])                                     # warm-up call, not timed
        lat = []
        for o in orig:
            t = time.perf_counter(); m(y, o); lat.append((time.perf_counter() - t) * 1000)
        out["models"][name] = {"p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1)}
        print(name, out["models"][name], flush=True)
    R.save("latency_idle", out)


if __name__ == "__main__":
    main()
