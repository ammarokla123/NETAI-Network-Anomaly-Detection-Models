"""Feature engineering pipeline for network telemetry data.

Transforms raw telemetry into model-ready feature tensors with rolling
statistics, lag features, and normalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

METRIC_COLUMNS = [
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "retransmits",
    "jitter_ms",
]


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std features for the given windows."""
    columns = columns or METRIC_COLUMNS
    windows = windows or [5, 15, 30]

    df = df.copy()
    for col in columns:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(window=w, min_periods=1).std().fillna(0)
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lag features (previous time-step values)."""
    columns = columns or METRIC_COLUMNS
    lags = lags or [1, 3, 5, 10]

    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag).bfill()
    return df


def add_rate_of_change(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add first-order difference (rate of change) features."""
    columns = columns or METRIC_COLUMNS
    df = df.copy()
    for col in columns:
        df[f"{col}_diff"] = df[col].diff().fillna(0)
    return df


def get_scaler(name: str = "standard") -> StandardScaler | MinMaxScaler | RobustScaler:
    """Return a scikit-learn scaler by name."""
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    return scalers[name]()


class FeaturePipeline:
    """End-to-end feature engineering pipeline.

    Usage::

        pipeline = FeaturePipeline(config)
        X_train, y_train = pipeline.fit_transform(train_df)
        X_val, y_val = pipeline.transform(val_df)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        fe_cfg = config.get("feature_engineering", {})
        self.rolling_windows: list[int] = fe_cfg.get("rolling_windows", [5, 15, 30])
        self.lag_steps: list[int] = fe_cfg.get("lag_steps", [1, 3, 5, 10])
        self.scaler_name: str = fe_cfg.get("scaler", "standard")
        self.normalize: bool = fe_cfg.get("normalize", True)
        self.scaler = get_scaler(self.scaler_name) if self.normalize else None
        self._feature_columns: list[str] | None = None

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = add_rolling_features(df, windows=self.rolling_windows)
        df = add_lag_features(df, lags=self.lag_steps)
        df = add_rate_of_change(df)
        return df

    @property
    def feature_columns(self) -> list[str]:
        if self._feature_columns is None:
            raise RuntimeError("Pipeline not yet fitted. Call fit_transform first.")
        return self._feature_columns

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Identify numeric feature columns (excluding metadata and labels)."""
        exclude = {"id", "timestamp", "source", "destination", "is_anomaly", "anomaly_type"}
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the pipeline on training data and return (features, labels)."""
        df = self._engineer(df)
        self._feature_columns = self._get_feature_columns(df)

        X = df[self._feature_columns].values.astype(np.float32)
        y = df["is_anomaly"].values.astype(np.float32) if "is_anomaly" in df.columns else None

        if self.scaler is not None:
            X = self.scaler.fit_transform(X).astype(np.float32)

        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform new data using the already-fitted pipeline."""
        if self._feature_columns is None:
            raise RuntimeError("Pipeline not yet fitted. Call fit_transform first.")

        df = self._engineer(df)
        X = df[self._feature_columns].values.astype(np.float32)
        y = df["is_anomaly"].values.astype(np.float32) if "is_anomaly" in df.columns else None

        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)

        return X, y
