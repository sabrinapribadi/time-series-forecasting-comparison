"""Rolling 24-hour benchmark (ADR-008): origins, scores and the Diebold–Mariano test. numpy/scipy only.

Rows are positions in the raw ETTh1 file (17,420 hourly rows). An origin is the first forecast hour; a model may use
OT values at rows < origin only and forecasts rows origin … origin + 23.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ETT" / "ETTh1.csv"
OUT_DIR = ROOT / "data" / "forecasts" / "ETT" / "rolling"
H = 24
N_ROWS = 17_420
TRAIN_END = 12_194                     # rows 0 … 12,193 are the training split (ADR-003)
TEST_START = 13_936
VAL_START = 12_218                     # ADR-008 correction: after ML training rows (end 12,200)
CLEAN_FROM = N_ROWS - 1_008            # no GIFT-Eval training portion reaches rows ≥ 16,412 (ADR-008 §3)
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def origins(split: str) -> list[int]:
    if split == "test":
        return [TEST_START + H * k for k in range(145)]
    if split == "val":
        return [VAL_START + H * k for k in range(71)]
    raise ValueError(split)


def clean_mask(orig: list[int]) -> np.ndarray:
    return np.array([o >= CLEAN_FROM for o in orig])


def load_ot() -> tuple[np.ndarray, np.ndarray]:
    """(dates as numpy datetime64, OT values) for all rows, in file order."""
    import csv
    with RAW.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == N_ROWS, len(rows)
    return np.array([r["date"] for r in rows], dtype="datetime64[s]"), np.array([float(r["OT"]) for r in rows])


def truth(y: np.ndarray, orig: list[int]) -> np.ndarray:
    return np.stack([y[o:o + H] for o in orig])


def mase_scale(y: np.ndarray) -> float:
    """In-sample MAE of 'same hour yesterday' on the training split (Hyndman & Koehler 2006)."""
    tr = y[:TRAIN_END]
    return float(np.mean(np.abs(tr[H:] - tr[:-H])))


def wql(y: np.ndarray, q: dict[str, np.ndarray]) -> float:
    """Weighted quantile loss over the given quantiles: 2 × Σ pinball / Σ |y| (averaged over quantiles)."""
    tot = 0.0
    for k, pred in q.items():
        a = float(k)
        e = y - pred
        tot += 2 * np.sum(np.maximum(a * e, (a - 1) * e))
    return float(tot / len(q) / np.sum(np.abs(y)))


def scores(y_true: np.ndarray, y_pred: np.ndarray, scale: float, q: dict | None = None) -> dict:
    e = y_true - y_pred
    out = {"mase": float(np.mean(np.abs(e)) / scale), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
           "per_origin_mae": np.mean(np.abs(e), axis=1).tolist()}
    if q:
        out["wql"] = wql(y_true, {k: np.asarray(v) for k, v in q.items()})
    return out


def dm_hln(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Diebold–Mariano test on loss differentials d = a − b, with the Harvey–Leybourne–Newbold small-sample correction.
    h = 1: per-origin losses come from non-overlapping 24-hour windows. Two-sided p from Student t (T − 1 df)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(hln), df=T - 1)
    return {"mean_diff": float(dbar), "stat": float(hln), "p": float(p), "n": T}


def save(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.json"
    p.write_text(json.dumps(payload))
    return p
