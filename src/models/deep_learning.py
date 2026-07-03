"""
Deep learning forecasting models: LSTM, Transformer, N-BEATS, TFT.

LSTM is implemented in pure PyTorch with:
  - StandardScaler normalization (fit on train, applied to val/test)
  - Early stopping (patience=15 on validation MSE)
  - ReduceLROnPlateau scheduler (patience=5, factor=0.5)
  - v6 defaults: input_len=168 (7-day window), hidden=256, layers=3

Transformer, N-BEATS, and TFT use the Darts library with:
  - EarlyStopping callback (patience=15 on val_loss)
  - 100 max epochs (typically stops at 30-60)
  - v6 defaults: input_chunk_length=168 for all Darts models
  - Transformer v6: d_model=128 (was 64)
  - TFT v6: hidden_size=128 (was 64); accepts past_covariates for multivariate conditioning

All Darts models accept a Darts TimeSeries object; helpers below handle
the pandas -> TimeSeries conversion.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: convert pandas Series -> Darts TimeSeries
# ---------------------------------------------------------------------------

def _to_darts(series: pd.Series | np.ndarray, dates: pd.DatetimeIndex | None = None):
    from darts import TimeSeries

    if isinstance(series, np.ndarray):
        if dates is None:
            raise ValueError("Pass dates when series is a numpy array")
        s = pd.Series(series.astype(np.float32), index=pd.DatetimeIndex(dates))
    else:
        s = series.astype(np.float32)
    return TimeSeries.from_series(s)


def _darts_early_stopping_kwargs(patience: int = 15):
    """Return pl_trainer_kwargs with EarlyStopping on val_loss."""
    try:
        from pytorch_lightning.callbacks import EarlyStopping
        cb = EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=False)
        return {"callbacks": [cb], "enable_progress_bar": False}
    except Exception:
        return {"enable_progress_bar": False}


# ---------------------------------------------------------------------------
# LSTM (PyTorch)
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    """Module-level LSTM net — must be at module scope so joblib can pickle it."""

    def __init__(self, hidden: int, layers: int, dropout: float, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMModel:
    """
    Vanilla LSTM regressor for direct multi-step forecasting.

    Trained to predict `horizon` future steps from a fixed-size
    input window of `input_len` past observations.

    Training improvements vs v1:
      - StandardScaler normalization (fit on train, applied to val/test)
      - Early stopping: patience=15 on validation MSE
      - ReduceLROnPlateau: patience=5, factor=0.5
      - Default 100 epochs (early stopping typically fires at 20-50)

    v6 capacity increase (underfitting remediation):
      - input_len: 96 → 168 (7-day window captures weekly seasonality)
      - hidden_size: 128 → 256
      - num_layers: 2 → 3
    """

    def __init__(
        self,
        input_len: int = 168,
        horizon: int = 24,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 15,
        device: str = "auto",
    ):
        self.input_len = input_len
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._net = None
        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        return (y - self._scaler_mean) / self._scaler_std

    def _denormalize(self, y: np.ndarray) -> np.ndarray:
        return y * self._scaler_std + self._scaler_mean

    def fit(self, y_train: np.ndarray, y_val: np.ndarray | None = None) -> "LSTMModel":
        from torch.utils.data import DataLoader, TensorDataset

        # Fit scaler on train
        self._scaler_mean = float(y_train.mean())
        self._scaler_std = float(y_train.std()) or 1.0
        y_tr = self._normalize(y_train)
        y_v = self._normalize(y_val) if y_val is not None else None

        device = torch.device(self.device)

        def _make_dataset(y):
            X, Y = [], []
            for i in range(len(y) - self.input_len - self.horizon + 1):
                X.append(y[i: i + self.input_len])
                Y.append(y[i + self.input_len: i + self.input_len + self.horizon])
            return (
                torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.array(Y), dtype=torch.float32),
            )

        X_tr, Y_tr = _make_dataset(y_tr)
        loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=self.batch_size, shuffle=True)

        self._net = _LSTMNet(self.hidden_size, self.num_layers, self.dropout, self.horizon).to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self._net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(loader)

            # Validation + early stopping — batched to avoid MPS OOM on large val sets
            val_loss = train_loss
            if y_v is not None and len(y_v) > self.input_len + self.horizon:
                X_v, Y_v = _make_dataset(y_v)
                self._net.eval()
                with torch.no_grad():
                    val_losses = []
                    for vi in range(0, len(X_v), self.batch_size):
                        xb = X_v[vi: vi + self.batch_size].to(device)
                        yb = Y_v[vi: vi + self.batch_size].to(device)
                        val_losses.append(loss_fn(self._net(xb), yb).item())
                    val_loss = float(np.mean(val_losses))
                self._net.train()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"LSTM epoch {epoch+1}/{self.epochs} "
                    f"train={train_loss:.4f} val={val_loss:.4f} "
                    f"lr={opt.param_groups[0]['lr']:.2e}"
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"LSTM early stopping at epoch {epoch+1} (best val={best_val_loss:.4f})")
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        return self

    def predict(self, y_context: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() before predict()")
        y_norm = self._normalize(y_context)
        self._net.eval()
        x = torch.tensor(y_norm[-self.input_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = self._net(x.to(self.device)).cpu().numpy().squeeze()
        return self._denormalize(pred_norm)


# ---------------------------------------------------------------------------
# Darts-based models (Transformer, N-BEATS, TFT)
# ---------------------------------------------------------------------------

class TransformerModel:
    """Attention-based Transformer for time series via Darts."""

    def __init__(
        self,
        input_chunk_length: int = 168,
        output_chunk_length: int = 24,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        from darts.models import TransformerModel as _DartsTransformer

        self._model = _DartsTransformer(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=random_state,
            pl_trainer_kwargs=_darts_early_stopping_kwargs(patience=15),
        )

    def fit(
        self,
        y_train: np.ndarray,
        dates_train: pd.DatetimeIndex,
        y_val: np.ndarray | None = None,
        dates_val: pd.DatetimeIndex | None = None,
    ) -> "TransformerModel":
        ts_train = _to_darts(y_train, dates_train)
        ts_val = _to_darts(y_val, dates_val) if y_val is not None else None
        self._model.fit(ts_train, val_series=ts_val)
        logger.info("Transformer fitted")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self._model.predict(horizon).values().squeeze()


class NBEATSModel:
    """Neural Basis Expansion Analysis for Time Series (N-BEATS) via Darts."""

    def __init__(
        self,
        input_chunk_length: int = 168,
        output_chunk_length: int = 24,
        num_stacks: int = 30,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_widths: int = 256,
        n_epochs: int = 100,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        from darts.models import NBEATSModel as _DartsNBEATS

        self._model = _DartsNBEATS(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=random_state,
            pl_trainer_kwargs=_darts_early_stopping_kwargs(patience=15),
        )

    def fit(
        self,
        y_train: np.ndarray,
        dates_train: pd.DatetimeIndex,
        y_val: np.ndarray | None = None,
        dates_val: pd.DatetimeIndex | None = None,
    ) -> "NBEATSModel":
        ts_train = _to_darts(y_train, dates_train)
        ts_val = _to_darts(y_val, dates_val) if y_val is not None else None
        self._model.fit(ts_train, val_series=ts_val)
        logger.info("N-BEATS fitted")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self._model.predict(horizon).values().squeeze()


class TFTModel:
    """Temporal Fusion Transformer (TFT) via Darts.

    Accepts optional past_covariates (load columns: HUFL/HULL/MUFL/MULL/LUFL/LULL)
    for multivariate conditioning. Covariates are stored during fit() and reused
    in predict() automatically.
    """

    def __init__(
        self,
        input_chunk_length: int = 168,
        output_chunk_length: int = 24,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        from darts.models import TFTModel as _DartsTFT

        self._model = _DartsTFT(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=random_state,
            add_relative_index=True,
            pl_trainer_kwargs=_darts_early_stopping_kwargs(patience=15),
        )
        self._past_cov_ts = None  # populated in fit() when covariates are provided

    def fit(
        self,
        y_train: np.ndarray,
        dates_train: pd.DatetimeIndex,
        y_val: np.ndarray | None = None,
        dates_val: pd.DatetimeIndex | None = None,
        past_cov_arr: np.ndarray | None = None,
        past_cov_dates: pd.DatetimeIndex | None = None,
        past_cov_val_arr: np.ndarray | None = None,
    ) -> "TFTModel":
        from darts import TimeSeries as _DartsTS

        ts_train = _to_darts(y_train, dates_train)
        ts_val = _to_darts(y_val, dates_val) if y_val is not None else None

        ts_cov = None
        ts_val_cov = None
        if past_cov_arr is not None and past_cov_dates is not None:
            cov_df = pd.DataFrame(past_cov_arr.astype(np.float32),
                                  index=pd.DatetimeIndex(past_cov_dates))
            ts_cov = _DartsTS.from_dataframe(cov_df)
            self._past_cov_ts = ts_cov
            if past_cov_val_arr is not None:
                val_cov_df = pd.DataFrame(past_cov_val_arr.astype(np.float32),
                                          index=pd.DatetimeIndex(dates_val))
                ts_val_cov = _DartsTS.from_dataframe(val_cov_df)

        self._model.fit(ts_train, past_covariates=ts_cov,
                        val_series=ts_val, val_past_covariates=ts_val_cov)
        logger.info(f"TFT fitted (past_covariates={'yes' if ts_cov else 'no'})")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self._model.predict(horizon,
                                   past_covariates=self._past_cov_ts).values().squeeze()
