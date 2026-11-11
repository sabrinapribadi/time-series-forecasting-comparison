"""Score the rolling benchmark and apply the ADR-008 gate. numpy/scipy only.

    python scripts/rolling_report.py        # reads data/forecasts/ETT/rolling/<model>.json, writes summary.json

1. Scores every model on validation, test, and the clean tail (MASE, MAE, RMSE; quantile loss where available).
2. Champions: lowest validation MASE in each group, univariate track (chosen before reading test scores).
3. Gate: the foundation champion wins if its test MASE is ≥ 5% lower, Diebold–Mariano–HLN p < 0.05 on per-origin MAE,
   and its clean-tail MASE is also lower; the trained champion wins under the mirror conditions; otherwise no clear winner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import rolling as R  # noqa: E402

OLD = R.ROOT / "data" / "forecasts" / "ETT"
MARGIN = 0.05


def load_all() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(R.OUT_DIR.glob("*.json")) if p.stem not in ("summary", "latency_idle")}


def score(run: dict, y: np.ndarray, scale: float) -> dict:
    out = {}
    for split in ("val", "test"):
        orig = run[split]["origins"]; yt = R.truth(y, orig); yp = np.array(run[split]["y_pred"])
        q = run[split].get("quantiles")
        out[split] = R.scores(yt, yp, scale, {k: np.array(v) for k, v in q.items()} if q else None)
        if split == "test":
            m = R.clean_mask(orig)
            qc = {k: np.array(v)[m] for k, v in q.items()} if q else None
            out["clean_tail"] = R.scores(yt[m], yp[m], scale, qc)
            lat = run["test"].get("latency_ms") or []
            out["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))} if lat else None
    return out


def gate(fm: str, tr: str, s: dict) -> dict:
    a, b = s[fm], s[tr]
    rel = 1 - a["test"]["mase"] / b["test"]["mase"]                  # > 0: foundation model better
    dm = R.dm_hln(np.array(a["test"]["per_origin_mae"]), np.array(b["test"]["per_origin_mae"]))
    clean_fm_better = a["clean_tail"]["mase"] < b["clean_tail"]["mase"]
    if rel >= MARGIN and dm["p"] < 0.05 and clean_fm_better:
        verdict = "foundation model wins"
    elif -rel / (1 - rel) >= MARGIN and dm["p"] < 0.05 and not clean_fm_better:   # trained ≥ 5% lower than foundation
        verdict = "trained model wins"
    else:
        verdict = "no clear winner"
    return {"foundation_champion": fm, "trained_champion": tr, "test_mase": {fm: a["test"]["mase"], tr: b["test"]["mase"]},
            "foundation_lower_by": rel, "dm_hln": dm, "clean_tail_mase": {fm: a["clean_tail"]["mase"], tr: b["clean_tail"]["mase"]},
            "verdict": verdict}


def main() -> None:
    _, y = R.load_ot(); scale = R.mase_scale(y)
    runs = load_all()
    s = {k: score(v, y, scale) for k, v in runs.items()}
    meta = {k: {f: v.get(f) for f in ("group", "track", "source", "hf_id", "licence", "params")} for k, v in runs.items()}
    uni = {k for k, v in runs.items() if v["track"] == "univariate"}
    pick = lambda g: min((k for k in uni if runs[k]["group"] == g), key=lambda k: s[k]["val"]["mase"])
    fm, tr = pick("foundation"), pick("trained")
    g = gate(fm, tr, s)
    base = {b: s[b]["test"]["mase"] for b in ("seasonal_naive_24", "seasonal_naive_168") if b in s}
    old = {}
    for k in runs:
        f = OLD / f"{k}_h1.json"
        if f.exists():
            d = json.loads(f.read_text()); old[k] = {"old_rmse": d["metrics"]["RMSE"], "old_mode": d.get("mode"), "rolling_rmse": s[k]["test"]["rmse"]}
    summary = {"design": "ADR-008", "mase_scale": scale, "origins": {"val": 71, "test": 145, "clean_tail": int(R.clean_mask(R.origins("test")).sum())},
               "models": {k: {**meta[k], **s[k]} for k in runs}, "champions": {"foundation": fm, "trained": tr}, "gate": g,
               "champions_beat_baselines": {c: all(s[c]["test"]["mase"] < v for v in base.values()) for c in (fm, tr)},
               "old_vs_rolling": old}
    if "chronos2_cov" in s:                                  # TFT dropped (ADR-008 deviation): compared with its own univariate run
        summary["covariate_track"] = {"chronos2_cov": s["chronos2_cov"]["test"]["mase"], "chronos2_univariate": s["chronos2"]["test"]["mase"],
                                      "chronos2_cov_clean": s["chronos2_cov"]["clean_tail"]["mase"], "chronos2_clean": s["chronos2"]["clean_tail"]["mase"]}
    summary["dropped"] = {"transformer": "CPU retraining exceeded 2 hours (ADR-008 deviation)", "tft": "CPU retraining exceeded 2 hours (ADR-008 deviation)"}
    idle = R.OUT_DIR / "latency_idle.json"
    if idle.exists():
        summary["latency_idle"] = json.loads(idle.read_text())
    R.save("summary", summary)
    print(f"champions: foundation={fm}, trained={tr}; verdict: {g['verdict']}")
    rows = sorted(uni | {"tft", "chronos2_cov"} & set(runs), key=lambda k: s[k]["test"]["mase"])
    print(f"{'model':22} {'group':10} {'val MASE':>9} {'test MASE':>9} {'clean':>7} {'RMSE':>7} {'WQL':>6} {'p50 ms':>7}")
    for k in rows:
        t = s[k]; w = t["test"].get("wql"); lat = t["latency_ms"]["p50"] if t["latency_ms"] else float("nan")
        print(f"{k:22} {runs[k]['group']:10} {t['val']['mase']:9.3f} {t['test']['mase']:9.3f} {t['clean_tail']['mase']:7.3f} "
              f"{t['test']['rmse']:7.3f} {('%.3f' % w) if w is not None else '   —':>6} {lat:7.0f}")


if __name__ == "__main__":
    main()
