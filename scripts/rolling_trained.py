"""Trained models and baselines on the rolling 24-hour benchmark (ADR-008). Runs in the project's Poetry environment.

    python scripts/rolling_trained.py --models seasonal_naive_24 holt_winters arima ...

Each model sees OT rows < origin only and forecasts the next 24 hours:
- baselines: same hour yesterday / last week;
- statistical: refitted on all history at each origin with the ADR-004 settings (ARIMA keeps its selected order 1,0,1);
- ML: recursive with the univariate checkpoints (time features + OT lags 1, 2, 24); each prediction feeds the next lags;
- LSTM: its 96-hour input window; Darts models: predict 24 from the history series, no retraining;
- TFT (covariate track): plus the six load series up to the origin.
The "univariate" Random Forest and CatBoost checkpoints turned out to be 15-feature (multivariate) models, so both are
retrained univariate with their ADR-004 settings (no tuning). Writes data/forecasts/ETT/rolling/<model>.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.ett_loader import LOAD_COLS, ETTLoader  # noqa: E402
from src.evaluation import rolling as R  # noqa: E402

MODELS_DIR = ROOT / "models"
H = R.H


def frame() -> pd.DataFrame:
    df = ETTLoader("h1").load()
    assert len(df) == R.N_ROWS
    return ETTLoader._add_time_features(df)


TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]


# ------------------------------------------------------------------ forecasters: f(o) -> 24 values
def seasonal_naive(y, lag):
    return lambda o: y[o - lag:o - lag + H].copy()


def holt_winters(y, _df):
    from src.models.statistical import HoltWintersModel
    ck = joblib.load(MODELS_DIR / "holt_winters_h1.joblib")
    def f(o):
        m = HoltWintersModel(trend=ck.trend, seasonal=ck.seasonal, seasonal_periods=ck.seasonal_periods, damped_trend=ck.damped_trend)
        return np.asarray(m.fit(y[:o]).predict(H))
    return f


def arima(y, _df):
    from statsmodels.tsa.arima.model import ARIMA
    order = joblib.load(MODELS_DIR / "arima_h1.joblib")._model.order          # (1, 0, 1), chosen by auto_arima in ADR-004
    return lambda o: np.asarray(ARIMA(y[:o], order=order).fit().forecast(H))


def prophet(y, df):
    import logging
    from src.models.statistical import ProphetModel
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    return lambda o: np.asarray(ProphetModel().fit(y[:o], dates_train=df["date"].iloc[:o]).predict(H))


def _ml_model(name, df):
    if name in ("xgboost", "lightgbm"):
        m = joblib.load(MODELS_DIR / f"{name}_h1.joblib")
        return m, "checkpoint"
    tr, va, _ = ETTLoader("h1").get_splits(mode="univariate", add_time_features=True, add_lag_features=True)
    cols = TIME_COLS + ["OT_lag_1", "OT_lag_2", "OT_lag_24"]
    if name == "random_forest":
        from src.models.ml import RandomForestModel
        m = RandomForestModel(n_estimators=200).fit(tr[cols].values, tr["OT"].values)
    else:
        from src.models.ml import CatBoostModel
        m = CatBoostModel(iterations=300).fit(tr[cols].values, tr["OT"].values, X_val=va[cols].values, y_val=va["OT"].values)
    joblib.dump(m, MODELS_DIR / f"{name}_univariate_h1.joblib")
    return m, "retrained univariate (ADR-004 settings)"


def ml(name):
    def build(y, df):
        m, how = _ml_model(name, df)
        T = df[TIME_COLS].to_numpy()
        def f(o):
            hist = list(y[:o]); out = []
            for h in range(H):
                t = o + h
                x = np.concatenate([T[t], [hist[t - 1], hist[t - 2], hist[t - 24]]])[None, :]
                p = float(np.asarray(m.predict(x)).ravel()[0])
                out.append(p); hist.append(p)                       # recursive: the prediction becomes the next lag
            return np.array(out)
        f.source = how
        return f
    return build


def lstm(y, _df):
    m = joblib.load(MODELS_DIR / "lstm_h1.joblib")
    m.device = "cpu"
    return lambda o: np.asarray(m.predict(y[:o]))[:H]


def _train_darts(name, df):
    """The joblib checkpoints hold untrained shells (epochs_trained = 0), so retrain with the ADR-004 settings exactly as
    scripts/train_model.py does (100 epochs max, early stopping on validation, same rows), on CPU; save in Darts' own format."""
    import torch
    from src.models import deep_learning as dl
    torch.set_num_threads(2)
    path = MODELS_DIR / f"{name}_rolling_h1.pt"
    cls = {"transformer": dl.TransformerModel, "nbeats": dl.NBEATSModel, "tft": dl.TFTModel}[name]
    if path.exists():
        return type(cls(n_epochs=100)._model).load(str(path), map_location="cpu"), "retrained (ADR-004 settings), loaded"
    w = cls(n_epochs=100)
    w._model.trainer_params = {**w._model.trainer_params, "accelerator": "cpu", "enable_progress_bar": False}
    tr, va, _ = ETTLoader("h1").get_splits(mode="univariate", add_time_features=True, add_lag_features=False)
    kw = dict(dates_train=pd.DatetimeIndex(tr["date"].values), y_val=va["OT"].values, dates_val=pd.DatetimeIndex(va["date"].values))
    if name == "tft":
        trm, vam, _ = ETTLoader("h1").get_splits(mode="multivariate", add_time_features=False, add_lag_features=False)
        kw.update(past_cov_arr=trm[LOAD_COLS].values, past_cov_dates=pd.DatetimeIndex(trm["date"].values), past_cov_val_arr=vam[LOAD_COLS].values)
    t = time.perf_counter(); w.fit(tr["OT"].values, **kw)
    print(f"{name} retrained in {time.perf_counter() - t:.0f} s, {w._model.epochs_trained} epochs", flush=True)
    w._model.save(str(path))
    return w._model, f"retrained (ADR-004 settings, {w._model.epochs_trained} epochs)"


def darts(name):
    def build(y, df):
        from darts import TimeSeries
        m, how = _train_darts(name, df)
        m.trainer_params = {**m.trainer_params, "accelerator": "cpu", "enable_progress_bar": False, "logger": False}
        dates = pd.DatetimeIndex(df["date"])
        cov = TimeSeries.from_dataframe(pd.DataFrame(df[LOAD_COLS].to_numpy(np.float32), index=dates)) if name == "tft" else None
        def f(o):
            s = TimeSeries.from_series(pd.Series(y[:o].astype(np.float32), index=dates[:o]))
            kw = {"past_covariates": cov[:o]} if cov is not None else {}
            return m.predict(H, series=s, verbose=False, **kw).values().ravel()
        f.source = how
        return f
    return build


MODELS = {
    "seasonal_naive_24": ("baseline", lambda y, df: seasonal_naive(y, 24)),
    "seasonal_naive_168": ("baseline", lambda y, df: seasonal_naive(y, 168)),
    "holt_winters": ("trained", holt_winters), "arima": ("trained", arima), "prophet": ("trained", prophet),
    "random_forest": ("trained", ml("random_forest")), "xgboost": ("trained", ml("xgboost")),
    "lightgbm": ("trained", ml("lightgbm")), "catboost": ("trained", ml("catboost")),
    "lstm": ("trained", lstm), "transformer": ("trained", darts("transformer")), "nbeats": ("trained", darts("nbeats")),
    "tft": ("trained", darts("tft")),
}


def run(name: str, y: np.ndarray, df: pd.DataFrame) -> dict:
    group, build = MODELS[name]
    f = build(y, df)
    out = {"model": name, "group": group, "track": "covariate" if name == "tft" else "univariate", "point": "point forecast",
           "source": getattr(f, "source", "checkpoint (ADR-004)" if group == "trained" else "rule")}
    for split in ("val", "test"):
        orig, preds, lat = R.origins(split), [], []
        for o in orig:
            t = time.perf_counter(); p = np.asarray(f(o), dtype=float); lat.append((time.perf_counter() - t) * 1000)
            assert p.shape == (H,) and np.isfinite(p).all(), (name, o, p.shape)
            preds.append(p.tolist())
        out[split] = {"origins": orig, "y_pred": preds, "latency_ms": [round(v, 2) for v in lat]}
        print(f"{name} {split}: {len(orig)} origins, p50 {np.percentile(lat, 50):.0f} ms", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    df = frame(); y = df["OT"].to_numpy(float)
    for name in a.models:
        R.save(name, run(name, y, df))


if __name__ == "__main__":
    main()
