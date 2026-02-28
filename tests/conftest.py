"""Shared test fixtures for NETAI Anomaly Detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from netai_anomaly.data.features import FeaturePipeline
from netai_anomaly.data.generator import generate_telemetry, save_to_sqlite


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Generate a small telemetry DataFrame for testing."""
    return generate_telemetry(num_samples=500, anomaly_ratio=0.1, seed=123)


@pytest.fixture(scope="session")
def tmp_db(sample_df: pd.DataFrame) -> str:
    """Create a temporary SQLite database with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        save_to_sqlite(sample_df, db_path)
        yield db_path


@pytest.fixture(scope="session")
def pipeline() -> FeaturePipeline:
    """Create a default feature pipeline."""
    return FeaturePipeline({
        "feature_engineering": {
            "rolling_windows": [5],
            "lag_steps": [1, 3],
            "scaler": "standard",
            "normalize": True,
        }
    })


@pytest.fixture(scope="session")
def feature_data(sample_df: pd.DataFrame, pipeline: FeaturePipeline):
    """Return (X, y) from the feature pipeline."""
    return pipeline.fit_transform(sample_df)


@pytest.fixture
def base_model_config():
    """Return a minimal model config dict."""
    return {
        "model": {
            "name": "autoencoder",
            "input_dim": 10,
            "encoder_dims": [8],
            "latent_dim": 4,
            "decoder_dims": [8],
            "dropout": 0.0,
            "use_batch_norm": False,
        },
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "patience": 5,
            "gradient_clip": 1.0,
            "scheduler": "cosine",
            "checkpoint_dir": "checkpoints",
        },
        "data": {
            "sequence_length": 10,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "evaluation": {
            "threshold_percentile": 95,
        },
    }
