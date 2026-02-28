"""Tests for the training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from netai_anomaly.data.dataset import FlatDataset, TelemetryDataset
from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.lstm import LSTMAnomalyDetector
from netai_anomaly.training.trainer import Trainer
from netai_anomaly.training.utils import set_seed


class TestTrainer:
    @pytest.fixture
    def ae_setup(self, tmp_path):
        """Set up autoencoder with tiny datasets for fast testing."""
        set_seed(42)
        input_dim = 10
        config = {
            "model": {
                "name": "autoencoder",
                "input_dim": input_dim,
                "encoder_dims": [8],
                "latent_dim": 4,
                "decoder_dims": [8],
                "dropout": 0.0,
                "use_batch_norm": False,
            },
            "training": {
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "patience": 10,
                "gradient_clip": 1.0,
                "scheduler": "cosine",
                "checkpoint_dir": str(tmp_path / "ckpts"),
            },
            "evaluation": {"threshold_percentile": 95},
        }

        model = Autoencoder(config)
        X = np.random.randn(100, input_dim).astype(np.float32)
        y = np.zeros(100, dtype=np.float32)
        train_ds = FlatDataset(X[:70], y[:70])
        val_ds = FlatDataset(X[70:], y[70:])
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8)

        return model, config, train_loader, val_loader

    def test_fit_returns_history(self, ae_setup):
        model, config, train_loader, val_loader = ae_setup
        trainer = Trainer(model, config, device="cpu")
        history = trainer.fit(train_loader, val_loader)

        assert "train_losses" in history
        assert "val_losses" in history
        assert "best_val_loss" in history
        assert len(history["train_losses"]) == 3
        assert history["best_val_loss"] > 0

    def test_loss_decreases(self, ae_setup):
        model, config, train_loader, val_loader = ae_setup
        config["training"]["epochs"] = 10
        trainer = Trainer(model, config, device="cpu")
        history = trainer.fit(train_loader, val_loader)

        assert history["train_losses"][-1] < history["train_losses"][0]

    def test_checkpoint_saved(self, ae_setup, tmp_path):
        model, config, train_loader, val_loader = ae_setup
        trainer = Trainer(model, config, device="cpu")
        trainer.fit(train_loader, val_loader)

        ckpt_path = Path(config["training"]["checkpoint_dir"]) / "autoencoder_best.pt"
        assert ckpt_path.exists()

    def test_compute_threshold(self, ae_setup):
        model, config, train_loader, val_loader = ae_setup
        trainer = Trainer(model, config, device="cpu")
        trainer.fit(train_loader, val_loader)
        threshold = trainer.compute_threshold(val_loader, percentile=95)
        assert threshold > 0
        assert isinstance(threshold, float)


class TestSequenceTrainer:
    def test_lstm_training(self, tmp_path):
        set_seed(42)
        input_dim = 8
        seq_len = 10
        config = {
            "model": {
                "name": "lstm",
                "input_dim": input_dim,
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "use_attention": False,
            },
            "training": {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "patience": 10,
                "gradient_clip": 1.0,
                "scheduler": "cosine",
                "checkpoint_dir": str(tmp_path / "ckpts"),
            },
            "evaluation": {"threshold_percentile": 95},
        }

        model = LSTMAnomalyDetector(config)
        X = np.random.randn(50, input_dim).astype(np.float32)
        y = np.zeros(50, dtype=np.float32)
        train_ds = TelemetryDataset(X[:35], y[:35], seq_len=seq_len)
        val_ds = TelemetryDataset(X[35:], y[35:], seq_len=seq_len)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4)

        trainer = Trainer(model, config, device="cpu")
        history = trainer.fit(train_loader, val_loader)
        assert len(history["train_losses"]) == 3


class TestDatasets:
    def test_flat_dataset(self):
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        ds = FlatDataset(X, y)
        assert len(ds) == 50
        x_i, y_i = ds[0]
        assert x_i.shape == (10,)
        assert y_i.ndim == 0

    def test_telemetry_dataset(self):
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)
        ds = TelemetryDataset(X, y, seq_len=20)
        assert len(ds) == 81  # 100 - 20 + 1
        x_i, y_i = ds[0]
        assert x_i.shape == (20, 10)

    def test_telemetry_dataset_label_is_max(self):
        X = np.random.randn(30, 5).astype(np.float32)
        y = np.zeros(30, dtype=np.float32)
        y[5] = 1.0  # anomaly at step 5
        ds = TelemetryDataset(X, y, seq_len=10)
        # Window [0:10] includes step 5 -> label should be 1
        _, label = ds[0]
        assert label.item() == 1.0
        # Window [6:16] does not include step 5 -> label should be 0
        _, label2 = ds[6]
        assert label2.item() == 0.0
