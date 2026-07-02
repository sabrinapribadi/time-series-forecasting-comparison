#!/usr/bin/env python3
"""
Train a single forecasting model on an ETT variant and save the checkpoint.

Usage:
    python scripts/train_model.py --model xgboost --variant h1
    python scripts/train_model.py --model xgboost --variant h1 --mode multivariate
    python scripts/train_model.py --model catboost --variant h1 --tune
    python scripts/train_model.py --model holt_winters --variant h1 --tune
    python scripts/train_model.py --model lstm --variant h1 --epochs 50

Mode:
    univariate   (default) — model sees only OT history + time features + lags
    multivariate           — ML models also receive all 6 load columns as covariates
                             (mirrors Volume_WD_GB regressor from IOH congestion pipeline)

--tune triggers Optuna HPO before fitting (IOH AOP2026 pattern):
    holt_winters  — 15 trials, minimize RMSE on train-fit residuals
    random_forest — 10 trials, maximize R² on validation
    xgboost       — 50 trials, maximize R² on validation
    catboost      —  8 trials, maximize R² on validation
    lightgbm      — fixed IOH production params (no Optuna)

Available models:
    Statistical : holt_winters, arima, prophet
    ML          : random_forest, xgboost, lightgbm, catboost
    Deep        : lstm, transformer, nbeats, tft
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ett_loader import ETTLoader, ETTVariant
from src.evaluation.metrics import compute_all_metrics
from src.utils import setup_logging, save_results

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

STATISTICAL = {"holt_winters", "arima", "prophet"}
ML_MODELS = {"random_forest", "xgboost", "lightgbm", "catboost"}
DL_MODELS = {"lstm", "transformer", "nbeats", "tft"}
ALL_MODELS = STATISTICAL | ML_MODELS | DL_MODELS

# Models that support Optuna tuning via tune_with_optuna()
TUNABLE = {"holt_winters", "random_forest", "xgboost", "catboost"}


def build_model(name: str, args: argparse.Namespace):
    if name == "holt_winters":
        from src.models.statistical import HoltWintersModel
        return HoltWintersModel(seasonal_periods=args.seasonal_periods)
    elif name == "arima":
        from src.models.statistical import ARIMAModel
        return ARIMAModel(auto=True, seasonal=False)
    elif name == "prophet":
        from src.models.statistical import ProphetModel
        return ProphetModel()
    elif name == "random_forest":
        from src.models.ml import RandomForestModel
        return RandomForestModel(n_estimators=200)
    elif name == "xgboost":
        from src.models.ml import XGBoostModel
        return XGBoostModel(n_estimators=300)
    elif name == "lightgbm":
        from src.models.ml import LightGBMModel
        return LightGBMModel()   # uses IOH production params by default
    elif name == "catboost":
        from src.models.ml import CatBoostModel
        return CatBoostModel(iterations=300)
    elif name == "lstm":
        from src.models.deep_learning import LSTMModel
        return LSTMModel(epochs=args.epochs)
    elif name == "transformer":
        from src.models.deep_learning import TransformerModel
        return TransformerModel(n_epochs=args.epochs)
    elif name == "nbeats":
        from src.models.deep_learning import NBEATSModel
        return NBEATSModel(n_epochs=args.epochs)
    elif name == "tft":
        from src.models.deep_learning import TFTModel
        return TFTModel(n_epochs=args.epochs)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train a time series forecasting model on ETT")
    parser.add_argument("--model", "-m", required=True, choices=sorted(ALL_MODELS))
    parser.add_argument("--variant", "-v", default="h1", choices=["h1", "h2", "m1", "m2"])
    parser.add_argument(
        "--mode",
        default="univariate",
        choices=["univariate", "multivariate"],
        help=(
            "univariate: model sees only OT history (default). "
            "multivariate: ML models also receive all 6 load columns as covariates."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=f"Run Optuna HPO before fitting. Supported: {sorted(TUNABLE)}",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for DL models")
    parser.add_argument("--seasonal-periods", type=int, default=24,
                        help="Seasonal period for Holt-Winters (24=hourly day, 96=15-min day)")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (steps)")
    args = parser.parse_args()

    mode_tag = f"_{args.mode}" if args.mode == "multivariate" else ""
    tune_tag = "_tuned" if args.tune else ""
    run_name = f"{args.model}{mode_tag}{tune_tag}_{args.variant}"

    setup_logging(log_file=f"train_{run_name}.log")
    logger.info(
        f"Training {args.model} on ETT-{args.variant.upper()}, "
        f"mode={args.mode}, tune={args.tune}, horizon={args.horizon}"
    )

    if args.tune and args.model not in TUNABLE:
        logger.warning(
            f"--tune not supported for {args.model}. "
            f"Supported: {sorted(TUNABLE)}. Continuing without Optuna."
        )

    # Load data
    loader = ETTLoader(args.variant)
    use_lags = args.model in ML_MODELS
    # Multivariate only applies to ML models (stat/DL use their own interfaces)
    effective_mode = args.mode if args.model in ML_MODELS else "univariate"

    train_df, val_df, test_df = loader.get_splits(
        mode=effective_mode,
        add_time_features=True,
        add_lag_features=use_lags,
    )

    y_train = train_df["OT"].values
    y_val = val_df["OT"].values
    y_test = test_df["OT"].values

    model = build_model(args.model, args)

    # Fit (or tune then fit)
    if args.model in STATISTICAL:
        if args.tune and args.model == "holt_winters":
            logger.info("Running Optuna HPO for HoltWinters (15 trials)...")
            model.tune_with_optuna(y_train, n_trials=15)
        elif args.model == "prophet":
            model.fit(y_train, dates_train=train_df["date"])
        else:
            model.fit(y_train)

    elif args.model in ML_MODELS:
        feature_cols = [c for c in train_df.columns if c not in ("date", "OT")]
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        X_test = test_df[feature_cols].values

        if args.tune and args.model in TUNABLE:
            n_trials_map = {"random_forest": 10, "xgboost": 50, "catboost": 8}
            n_trials = n_trials_map.get(args.model, 20)
            logger.info(f"Running Optuna HPO for {args.model} ({n_trials} trials)...")
            model.tune_with_optuna(X_train, y_train, X_val, y_val, n_trials=n_trials)
        elif args.model in {"xgboost", "lightgbm", "catboost"}:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)

    elif args.model == "lstm":
        model.fit(y_train, y_val=y_val)

    else:
        # Darts models require DatetimeIndex
        model.fit(
            y_train,
            dates_train=pd.DatetimeIndex(train_df["date"].values),
            y_val=y_val,
            dates_val=pd.DatetimeIndex(val_df["date"].values),
        )

    # Predict
    if args.model in STATISTICAL:
        y_pred = model.predict(len(y_test))
    elif args.model in ML_MODELS:
        y_pred = model.predict(X_test)
    elif args.model == "lstm":
        # Rolling-window oracle evaluation: predict horizon steps, advance by horizon,
        # refresh context with ground-truth OT (consistent with ML oracle lag approach)
        context = np.concatenate([y_train, y_val])
        preds = []
        h = model.horizon
        for i in range(0, len(y_test), h):
            chunk = model.predict(context)
            preds.extend(chunk.tolist())
            context = np.append(context, y_test[i: i + h])
        y_pred = np.array(preds[: len(y_test)])
    else:
        # Darts models: predict full test length (autoregressive internally)
        y_pred = model.predict(len(y_test))

    # Evaluate
    metrics = compute_all_metrics(y_test, y_pred, y_train=y_train)
    logger.info(f"Test metrics: {metrics}")

    # Save results for dashboard first — before potentially slow checkpoint I/O
    out = save_results(
        {
            "model": args.model,
            "variant": args.variant,
            "mode": effective_mode,
            "tuned": args.tune and args.model in TUNABLE,
            "horizon": args.horizon,
            "metrics": metrics,
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        },
        name=run_name,
    )
    logger.info(f"Results saved to {out}")

    # Save checkpoint (move PyTorch nets to CPU first for portability)
    import joblib
    if args.model == "lstm" and hasattr(model, "_net") and model._net is not None:
        model._net = model._net.cpu()
    ckpt_path = MODELS_DIR / f"{run_name}.joblib"
    joblib.dump(model, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
