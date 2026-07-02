"""
Deep learning forecasting models: LSTM, Transformer, N-BEATS, TFT.

LSTM is implemented in pure PyTorch.
Transformer, N-BEATS, and TFT use the Darts library, which provides
battle-tested implementations with a unified probabilistic API.

All Darts models accept a Darts TimeSeries object; helpers below handle
the pandas → TimeSeries conversion.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: convert pandas Series → Darts TimeSeries
# ---------------------------------------------------------------------------

def _to_darts(series: pd.Series | np.ndarray, dates: pd.DatetimeIndex | None = None):
    from darts import TimeSeries

    if isinstance(series, np.ndarray):
        if dates is None:
            raise ValueError("Pass dates when series is a numpy array")
        s = pd.Series(series.astype(np.float32), index=pd.DatetimeIndex(dates))
    else:
        s = series.astype(np.float32)
    # Cast to float32 before construction — avoids MPS float64 error on Apple Silicon
    return TimeSeries.from_series(s)


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
    """

    def __init__(
        self,
        input_len: int = 96,
        horizon: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
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

        import torch
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

    def fit(self, y_train: np.ndarray, y_val: np.ndarray | None = None) -> "LSTMModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.device)

        def _make_dataset(y):
            X, Y = [], []
            for i in range(len(y) - self.input_len - self.horizon + 1):
                X.append(y[i: i + self.input_len])
                Y.append(y[i + self.input_len: i + self.input_len + self.horizon])
            return torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1), \
                   torch.tensor(np.array(Y), dtype=torch.float32)

        X_tr, Y_tr = _make_dataset(y_train)
        loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=self.batch_size, shuffle=True)

        self._net = _LSTMNet(self.hidden_size, self.num_layers, self.dropout, self.horizon).to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM epoch {epoch+1}/{self.epochs} loss={epoch_loss/len(loader):.4f}")

        return self

    def predict(self, y_context: np.ndarray) -> np.ndarray:
        import torch
        if self._net is None:
            raise RuntimeError("Call fit() before predict()")
        self._net.eval()
        x = torch.tensor(y_context[-self.input_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            return self._net(x.to(self.device)).cpu().numpy().squeeze()


# ---------------------------------------------------------------------------
# Darts-based models (Transformer, N-BEATS, TFT)
# ---------------------------------------------------------------------------

class TransformerModel:
    """Attention-based Transformer for time series via Darts."""

    def __init__(
        self,
        input_chunk_length: int = 96,
        output_chunk_length: int = 24,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        n_epochs: int = 30,
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
            pl_trainer_kwargs={"enable_progress_bar": False},
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
        input_chunk_length: int = 96,
        output_chunk_length: int = 24,
        num_stacks: int = 30,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_widths: int = 256,
        n_epochs: int = 30,
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
            pl_trainer_kwargs={"enable_progress_bar": False},
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
    """Temporal Fusion Transformer (TFT) via Darts."""

    def __init__(
        self,
        input_chunk_length: int = 96,
        output_chunk_length: int = 24,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        n_epochs: int = 30,
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
            add_relative_index=True,  # auto-generates future covariates from time index
            pl_trainer_kwargs={"enable_progress_bar": False},
        )

    def fit(
        self,
        y_train: np.ndarray,
        dates_train: pd.DatetimeIndex,
        y_val: np.ndarray | None = None,
        dates_val: pd.DatetimeIndex | None = None,
    ) -> "TFTModel":
        ts_train = _to_darts(y_train, dates_train)
        ts_val = _to_darts(y_val, dates_val) if y_val is not None else None
        self._model.fit(ts_train, val_series=ts_val)
        logger.info("TFT fitted")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self._model.predict(horizon).values().squeeze()
