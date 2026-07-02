#!/usr/bin/env python3
"""
Run inference with a saved model checkpoint and export results for the dashboard.

Usage:
    python scripts/run_inference.py --model xgboost --variant h1
    python scripts/run_inference.py --all --variant h1
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib

from src.data.ett_loader import ETTLoader
from src.evaluation.metrics import compute_all_metrics
from src.utils import setup_logging, save_results

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"

ML_MODELS = {"random_forest", "xgboost", "lightgbm", "catboost"}


def run_single(model_name: str, variant: str) -> dict:
    ckpt_path = MODELS_DIR / f"{model_name}_{variant}.joblib"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run: python scripts/train_model.py --model {model_name} --variant {variant}"
        )

    model = joblib.load(ckpt_path)
    loader = ETTLoader(variant)
    use_lags = model_name in ML_MODELS
    train_df, _, test_df = loader.get_splits(add_time_features=True, add_lag_features=use_lags)

    y_train = train_df["OT"].values
    y_test = test_df["OT"].values

    if model_name in ML_MODELS:
        feature_cols = [c for c in test_df.columns if c not in ("date", "OT")]
        y_pred = model.predict(test_df[feature_cols].values)
    else:
        y_pred = model.predict(len(y_test))
        y_pred = y_pred[: len(y_test)]

    metrics = compute_all_metrics(y_test, y_pred, y_train=y_train)
    logger.info(f"{model_name} ({variant}): {metrics}")

    result = {
        "model": model_name,
        "variant": variant,
        "metrics": metrics,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }
    out = save_results(result, name=f"{model_name}_{variant}")
    logger.info(f"Saved to {out}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run inference and export results for dashboard")
    parser.add_argument("--model", "-m", help="Model name (required unless --all)")
    parser.add_argument("--variant", "-v", default="h1", choices=["h1", "h2", "m1", "m2"])
    parser.add_argument("--all", action="store_true", help="Run inference for all saved checkpoints")
    args = parser.parse_args()

    setup_logging(log_file="inference.log")

    if args.all:
        checkpoints = list(MODELS_DIR.glob(f"*_{args.variant}.joblib"))
        if not checkpoints:
            logger.error(f"No checkpoints found for variant {args.variant} in {MODELS_DIR}")
            sys.exit(1)
        for ckpt in checkpoints:
            model_name = ckpt.stem.replace(f"_{args.variant}", "")
            run_single(model_name, args.variant)
    elif args.model:
        run_single(args.model, args.variant)
    else:
        parser.error("Provide --model <name> or --all")


if __name__ == "__main__":
    main()
