#!/usr/bin/env python3
"""
Extract feature importances from saved ML model checkpoints.

Each model is loaded in an isolated subprocess so that segfaults (library
version mismatches between save and load time) don't kill the main process.

Saves:  data/forecasts/ETT/{model}_{variant}_feature_importance.json
Usage:  PYTHONPATH=. poetry run python scripts/extract_feature_importances.py
"""

import json
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "forecasts" / "ETT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ML_MODELS = ["random_forest", "xgboost", "lightgbm", "catboost"]
VARIANTS = ["h1"]

FEAT_NAMES_UNI = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "OT_lag_1", "OT_lag_2", "OT_lag_24",
    "OT_rolling_mean_3", "OT_rolling_std_3", "OT_growth_rate", "OT_trend_3",
]
FEAT_NAMES_MV = FEAT_NAMES_UNI + ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


def extract_one(joblib_path: str) -> list[float] | None:
    """Run extraction in a subprocess; returns importances list or None on failure."""
    code = f"""
import sys, json
sys.path.insert(0, '.')
import joblib
m = joblib.load(r'{joblib_path}')
fi = getattr(m, 'feature_importances_', None)
if fi is None:
    inner = getattr(m, '_model', getattr(m, 'model', None))
    if inner is not None:
        fi = getattr(inner, 'feature_importances_', None)
if fi is not None:
    print(json.dumps([float(x) for x in fi]))
else:
    print('null')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        out = result.stdout.strip()
        parsed = json.loads(out)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


def main():
    print("Extracting feature importances from ML model checkpoints")
    print("=" * 60)

    for model in ML_MODELS:
        for variant in VARIANTS:
            run_key = f"{model}_{variant}"
            joblib_path = MODELS_DIR / f"{run_key}.joblib"
            out_path = OUTPUT_DIR / f"{run_key}_feature_importance.json"

            if not joblib_path.exists():
                print(f"[SKIP]  {run_key}.joblib — not found")
                continue

            print(f"[LOAD]  {run_key}.joblib ...", end=" ", flush=True)
            importances = extract_one(str(joblib_path))

            if importances is None:
                print("FAILED (segfault or load error)")
                continue

            n = len(importances)
            if n == len(FEAT_NAMES_MV):
                feat_names = FEAT_NAMES_MV
            elif n == len(FEAT_NAMES_UNI):
                feat_names = FEAT_NAMES_UNI
            else:
                feat_names = [f"feature_{i}" for i in range(n)]

            out = {
                "model": model,
                "variant": variant,
                "n_features": n,
                "feature_names": feat_names,
                "importances": importances,
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"OK ({n} features) -> {out_path.name}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
