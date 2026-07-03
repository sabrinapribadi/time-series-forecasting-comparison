"""
ETT (Electricity Transformer Temperature) dataset loader.

Handles loading, splitting, and feature engineering for all four ETT variants:
  ETTh1, ETTh2  — hourly (17,420 rows each, 2016-07-01 to 2018-06-26)
  ETTm1, ETTm2  — 15-minute (69,680 rows each, same date range)

Two modes (ADR-001):
  univariate   — target OT from its own history + time features + optional lags
  multivariate — adds all 6 load columns (HUFL, HULL, MUFL, MULL, LUFL, LULL)
                 as covariates; analogous to Volume_WD_GB regressor in IOH pipeline

Standard chronological split (per ADR-003):
  Train: 70% | Validation: 10% | Test: 20%
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "ETT"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "ETT"

COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET = "OT"
LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

# Chronological split ratios (ADR-003)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


class ETTVariant(str, Enum):
    H1 = "h1"
    H2 = "h2"
    M1 = "m1"
    M2 = "m2"

    @property
    def filename(self) -> str:
        return f"ETT{self.value}.csv"

    @property
    def frequency(self) -> str:
        return "h" if self.value.startswith("h") else "15min"


class ETTLoader:
    """Load and split an ETT variant, with optional feature engineering."""

    def __init__(self, variant: ETTVariant | str = ETTVariant.H1):
        self.variant = ETTVariant(variant) if isinstance(variant, str) else variant
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Return the full dataset with datetime index."""
        if self._df is None:
            path = RAW_DIR / self.variant.filename
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run: python scripts/download_data.py"
                )
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self._df = df
        return self._df.copy()

    def get_splits(
        self,
        mode: str = "univariate",
        add_time_features: bool = True,
        add_lag_features: bool = False,
        add_rolling_features: bool = False,
        lag_steps: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return (train, val, test) DataFrames with chronological split.

        Args:
            mode:                 'univariate' — OT target only; drops raw load cols.
                                  'multivariate' — keeps HUFL/HULL/MUFL/MULL/LUFL/LULL
                                  as covariates (mirrors Volume_WD_GB regressor from
                                  IOH congestion pipeline).
            add_time_features:    Cyclical sin/cos encodings for hour, day-of-week,
                                  month (ADR-002).
            add_lag_features:     Autoregressive lag columns for tree-based models
                                  (ADR-005). Only added for the target column.
            add_rolling_features: Rolling mean/std/trend/growth_rate features shifted
                                  by 1 step (from AOP2026 feature engineering).
            lag_steps:            Which lags to compute. Defaults to [1, 2, 24] for
                                  hourly variants, [1, 4, 96] for 15-min variants.
        """
        if mode not in ("univariate", "multivariate"):
            raise ValueError(f"mode must be 'univariate' or 'multivariate', got {mode!r}")

        df = self.load()

        if add_time_features:
            df = self._add_time_features(df)

        if add_lag_features:
            steps = lag_steps or self._default_lags()
            df = self._add_lag_features(df, steps)

        if add_rolling_features:
            df = self._add_rolling_features(df)

        if add_lag_features or add_rolling_features:
            df = df.dropna().reset_index(drop=True)

        # In univariate mode, drop the 6 raw load columns — model only sees OT history
        if mode == "univariate":
            df = df.drop(columns=LOAD_COLS, errors="ignore")

        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()

        return train, val, test

    def get_arrays(
        self,
        feature_cols: list[str] | None = None,
        **split_kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Return X/y numpy arrays for each split.

        Returns:
            {
              "train": {"X": ..., "y": ...},
              "val":   {"X": ..., "y": ...},
              "test":  {"X": ..., "y": ...},
            }
        """
        train, val, test = self.get_splits(**split_kwargs)

        if feature_cols is None:
            feature_cols = [c for c in train.columns if c not in ("date", TARGET)]

        result = {}
        for name, df in [("train", train), ("val", val), ("test", test)]:
            result[name] = {
                "X": df[feature_cols].values.astype(np.float32),
                "y": df[TARGET].values.astype(np.float32),
                "dates": df["date"].values,
            }
        return result

    # ------------------------------------------------------------------
    # Feature engineering helpers (ADR-002)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        dt = df["date"].dt
        df = df.copy()

        # Cyclical hour encoding (period = 24 for hourly, 96 for 15-min)
        hour_frac = dt.hour + dt.minute / 60
        df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

        # Cyclical day-of-week (period = 7)
        df["dow_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

        # Cyclical month (period = 12)
        df["month_sin"] = np.sin(2 * np.pi * (dt.month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (dt.month - 1) / 12)

        return df

    def _add_lag_features(
        self, df: pd.DataFrame, lag_steps: list[int]
    ) -> pd.DataFrame:
        df = df.copy()
        for lag in lag_steps:
            df[f"OT_lag_{lag}"] = df[TARGET].shift(lag)
        return df

    @staticmethod
    def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """Rolling stats on OT, all shifted by 1 step to prevent leakage (from AOP2026)."""
        df = df.copy()
        ot = df[TARGET]
        df["OT_rolling_mean_3"] = ot.rolling(3).mean().shift(1)
        df["OT_rolling_std_3"] = ot.rolling(3).std().shift(1)
        df["OT_growth_rate"] = ot.pct_change().shift(1)
        df["OT_trend_3"] = (
            ot.rolling(3)
            .apply(lambda x: np.polyfit(range(3), x, 1)[0], raw=True)
            .shift(1)
        )
        return df

    def _default_lags(self) -> list[int]:
        if self.variant.frequency == "h":
            return [1, 2, 24]
        return [1, 4, 96]
