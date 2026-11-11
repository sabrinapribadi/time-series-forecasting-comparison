"""Zero-shot foundation models on the rolling 24-hour benchmark (ADR-008). CPU only, no training on ETTh1.

    ~/.venvs/tsfc-fm/bin/python scripts/rolling_foundation.py [--models chronos2 timesfm25 moirai2 chronos2_cov]

Each origin is forecast on its own (batch of 1) so the latency is what one request costs. Context: the last 2,048 hours
before the origin. Point forecast: the median (0.5 quantile) for every model; quantiles 0.1–0.9 kept for the quantile loss.
Writes data/forecasts/ETT/rolling/<model>.json.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import rolling as R  # noqa: E402

CTX = 2_048
LOADS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def loads() -> dict[str, np.ndarray]:
    with R.RAW.open() as f:
        rows = list(csv.DictReader(f))
    return {c: np.array([float(r[c]) for r in rows]) for c in LOADS}


class Chronos2:
    hf, licence, params = "amazon/chronos-2", "Apache-2.0", "120M"

    def __init__(self, covariates: bool = False):
        from chronos import Chronos2Pipeline
        self.p = Chronos2Pipeline.from_pretrained(self.hf, device_map="cpu")
        self.cov = loads() if covariates else None

    def __call__(self, y: np.ndarray, o: int) -> np.ndarray:
        lo = max(0, o - CTX)
        inp = {"target": y[lo:o]}
        if self.cov is not None:
            inp["past_covariates"] = {c: v[lo:o] for c, v in self.cov.items()}
        q, _ = self.p.predict_quantiles([inp], prediction_length=R.H, quantile_levels=QS)
        return q[0].squeeze(0).numpy().T                      # (9 quantiles, 24)


class TimesFM25:
    hf, licence, params = "google/timesfm-2.5-200m-pytorch", "Apache-2.0", "200M"

    def __init__(self):
        import timesfm
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.hf)
        self.m.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=128, normalize_inputs=True, use_continuous_quantile_head=True,
                                              force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))

    def __call__(self, y: np.ndarray, o: int) -> np.ndarray:
        _, q = self.m.forecast(horizon=R.H, inputs=[y[max(0, o - CTX):o]])
        return q[0, :, 1:].T                                   # drop the mean column → (9 quantiles 0.1…0.9, 24)


class Moirai2:
    hf, licence, params = "Salesforce/moirai-2.0-R-small", "CC BY-NC 4.0", "11M"

    def __init__(self):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        self.m = Moirai2Forecast(module=Moirai2Module.from_pretrained(self.hf), prediction_length=R.H, context_length=CTX,
                                 target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        ql = list(getattr(self.m.module, "quantile_levels", QS))
        self.idx = [ql.index(q) for q in QS]

    def __call__(self, y: np.ndarray, o: int) -> np.ndarray:
        out = self.m.predict(past_target=[y[max(0, o - CTX):o, None]])     # (1, n_quantiles, 24)
        return np.asarray(out)[0][self.idx].reshape(len(QS), R.H)


MODELS = {"chronos2": lambda: Chronos2(), "chronos2_cov": lambda: Chronos2(covariates=True), "timesfm25": TimesFM25, "moirai2": Moirai2}


def run(name: str, y: np.ndarray) -> dict:
    t0 = time.perf_counter(); model = MODELS[name](); load_s = time.perf_counter() - t0
    out = {"model": name, "group": "foundation", "track": "covariate" if name.endswith("_cov") else "univariate",
           "hf_id": model.hf, "licence": model.licence, "params": model.params, "context": CTX, "point": "median",
           "load_s": round(load_s, 1), "cpu": platform.processor() or platform.machine(), "torch_threads": torch.get_num_threads()}
    for split in ("val", "test"):
        orig, preds, quant, lat = R.origins(split), [], {str(q): [] for q in QS}, []
        for o in orig:
            t = time.perf_counter(); qf = model(y, o); lat.append((time.perf_counter() - t) * 1000)
            assert qf.shape == (len(QS), R.H) and np.isfinite(qf).all(), (name, o, qf.shape)
            preds.append(qf[QS.index(0.5)].tolist())
            for i, q in enumerate(QS):
                quant[str(q)].append(qf[i].tolist())
        out[split] = {"origins": orig, "y_pred": preds, "quantiles": quant, "latency_ms": [round(v, 2) for v in lat]}
        print(f"{name} {split}: {len(orig)} origins, p50 {np.percentile(lat, 50):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    _, y = R.load_ot()
    for name in a.models:
        R.save(name, run(name, y))


if __name__ == "__main__":
    main()
