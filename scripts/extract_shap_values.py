#!/usr/bin/env python3
"""
Compute SHAP values for saved ML model checkpoints.

Each model is loaded in an isolated subprocess to prevent segfaults
(XGBoost and LightGBM have library version mismatches).

Feature matrix is reconstructed from the raw ETTh1 data using the exact same
pipeline that was used during training:
  mode=multivariate, add_time_features=True, add_lag_features=True
  → 15 features: HUFL, HULL, MUFL, MULL, LUFL, LULL,
                  hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
                  OT_lag_1, OT_lag_2, OT_lag_24

Saves: data/forecasts/ETT/{model}_{variant}_shap.json
Usage: PYTHONPATH=. poetry run python scripts/extract_shap_values.py
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "forecasts" / "ETT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ML_MODELS = ["random_forest", "xgboost", "lightgbm", "catboost"]
VARIANTS = ["h1"]
N_BACKGROUND = 100   # background samples for TreeExplainer
N_EXPLAIN = 200      # test samples to explain


def extract_shap_one(joblib_path: str, project_root: str, variant: str) -> dict | None:
    code = f"""
import os, sys, json
# Must precede any torch/xgboost/lgbm import to avoid macOS threading segfaults
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, r'{project_root}')
import numpy as np
import joblib
import shap

# Reconstruct the exact 15-feature test matrix used during training
from src.data.ett_loader import ETTLoader
loader = ETTLoader('{variant}')
train_df, val_df, test_df = loader.get_splits(
    mode='multivariate',
    add_time_features=True,
    add_lag_features=True,
)
feat_cols = [c for c in train_df.columns if c not in ('date', 'OT')]
X_train = train_df[feat_cols].values.astype('float32')
X_test  = test_df[feat_cols].values.astype('float32')

# Load model — always unwrap the custom wrapper to get the sklearn/boosting estimator
m = joblib.load(r'{joblib_path}')
estimator = getattr(m, '_model', getattr(m, 'model', m))

# Background: stratified subsample of training data
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X_train), size=min({N_BACKGROUND}, len(X_train)), replace=False)
background = X_train[bg_idx]

# Test subset to explain
te_idx = rng.choice(len(X_test), size=min({N_EXPLAIN}, len(X_test)), replace=False)
te_idx = np.sort(te_idx)
X_explain = X_test[te_idx]

# Compute SHAP values
explainer = shap.TreeExplainer(estimator, data=background, feature_perturbation='interventional')
shap_vals = explainer.shap_values(X_explain)   # shape (N_EXPLAIN, 15)

mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
# Save a compact sample (every 5th row) for beeswarm
sample_shap  = shap_vals[::5].tolist()
sample_feats = X_explain[::5].tolist()

result = {{
    'feature_names': feat_cols,
    'mean_abs_shap': mean_abs,
    'shap_sample': sample_shap,
    'feature_sample': sample_feats,
    'n_explained': int(len(X_explain)),
    'n_background': int(len(background)),
}}
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            if result.stderr:
                print(f"    stderr: {result.stderr[-300:]}")
            return None
        out = result.stdout.strip()
        if not out:
            return None
        return json.loads(out)
    except Exception as e:
        print(f"    exception: {e}")
        return None


def main():
    print("Computing SHAP values for ML model checkpoints")
    print("=" * 60)

    for model in ML_MODELS:
        for variant in VARIANTS:
            run_key = f"{model}_{variant}"
            joblib_path = MODELS_DIR / f"{run_key}.joblib"
            out_path = OUTPUT_DIR / f"{run_key}_shap.json"

            if not joblib_path.exists():
                print(f"[SKIP]  {run_key}.joblib — not found")
                continue

            print(f"[SHAP]  {run_key} ...", end=" ", flush=True)
            data = extract_shap_one(str(joblib_path), str(PROJECT_ROOT), variant)

            if data is None:
                print("FAILED (segfault or load error)")
                continue

            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            top = sorted(zip(data["feature_names"], data["mean_abs_shap"]),
                         key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{n}={v:.4f}" for n, v in top)
            print(f"OK  top3: {top_str}  -> {out_path.name}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
