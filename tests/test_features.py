"""Tests for the feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netai_anomaly.data.features import (
    FeaturePipeline,
    add_lag_features,
    add_rate_of_change,
    add_rolling_features,
)
from netai_anomaly.data.generator import generate_telemetry


@pytest.fixture
def small_df():
    return generate_telemetry(num_samples=200, seed=99)


class TestRollingFeatures:
    def test_adds_columns(self, small_df):
        result = add_rolling_features(small_df, windows=[5])
        assert "throughput_mbps_roll_mean_5" in result.columns
        assert "throughput_mbps_roll_std_5" in result.columns

    def test_no_nans(self, small_df):
        result = add_rolling_features(small_df, windows=[5, 10])
        for col in result.columns:
            if "roll_" in col:
                assert not result[col].isna().any(), f"NaN found in {col}"


class TestLagFeatures:
    def test_adds_columns(self, small_df):
        result = add_lag_features(small_df, lags=[1, 3])
        assert "throughput_mbps_lag_1" in result.columns
        assert "latency_ms_lag_3" in result.columns

    def test_no_nans(self, small_df):
        result = add_lag_features(small_df, lags=[1])
        for col in result.columns:
            if "lag_" in col:
                assert not result[col].isna().any(), f"NaN found in {col}"


class TestRateOfChange:
    def test_adds_diff_columns(self, small_df):
        result = add_rate_of_change(small_df)
        assert "throughput_mbps_diff" in result.columns
        assert "latency_ms_diff" in result.columns


class TestFeaturePipeline:
    def test_fit_transform_shape(self, small_df):
        pipeline = FeaturePipeline({
            "feature_engineering": {
                "rolling_windows": [5],
                "lag_steps": [1],
                "scaler": "standard",
                "normalize": True,
            }
        })
        X, y = pipeline.fit_transform(small_df)
        assert X.ndim == 2
        assert X.shape[0] == len(small_df)
        assert y is not None
        assert len(y) == len(small_df)

    def test_transform_after_fit(self, small_df):
        pipeline = FeaturePipeline()
        X_train, _ = pipeline.fit_transform(small_df)
        X_test, _ = pipeline.transform(small_df)
        assert X_train.shape == X_test.shape

    def test_transform_before_fit_raises(self, small_df):
        pipeline = FeaturePipeline()
        with pytest.raises(RuntimeError, match="not yet fitted"):
            pipeline.transform(small_df)

    def test_num_features(self, small_df):
        pipeline = FeaturePipeline()
        pipeline.fit_transform(small_df)
        assert pipeline.num_features > 5  # should have more than raw features

    def test_no_nans_in_output(self, small_df):
        pipeline = FeaturePipeline()
        X, _ = pipeline.fit_transform(small_df)
        assert not np.isnan(X).any()

    def test_normalized_distribution(self, small_df):
        pipeline = FeaturePipeline({
            "feature_engineering": {"normalize": True, "scaler": "standard"}
        })
        X, _ = pipeline.fit_transform(small_df)
        # Standard scaler: mean ~0, std ~1
        assert abs(X.mean()) < 0.5
        assert abs(X.std() - 1.0) < 0.5

    def test_minmax_scaler(self, small_df):
        pipeline = FeaturePipeline({
            "feature_engineering": {"normalize": True, "scaler": "minmax"}
        })
        X, _ = pipeline.fit_transform(small_df)
        assert X.min() >= -0.01
        assert X.max() <= 1.01

    def test_no_normalization(self, small_df):
        pipeline = FeaturePipeline({
            "feature_engineering": {"normalize": False}
        })
        X, _ = pipeline.fit_transform(small_df)
        # Raw values should have large range (throughput in thousands)
        assert X.max() > 100
