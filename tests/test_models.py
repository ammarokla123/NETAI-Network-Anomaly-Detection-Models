"""Tests for all anomaly detection model architectures."""

from __future__ import annotations

import pytest
import torch

from netai_anomaly.models.autoencoder import Autoencoder
from netai_anomaly.models.base import create_model, register_model, _MODEL_REGISTRY
from netai_anomaly.models.lstm import LSTMAnomalyDetector
from netai_anomaly.models.transformer import TransformerAnomalyDetector


class TestAutoencoder:
    @pytest.fixture
    def config(self):
        return {
            "model": {
                "name": "autoencoder",
                "input_dim": 20,
                "encoder_dims": [16, 8],
                "latent_dim": 4,
                "decoder_dims": [8, 16],
                "dropout": 0.1,
                "use_batch_norm": True,
            }
        }

    def test_forward_shape(self, config):
        model = Autoencoder(config)
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 20)

    def test_encode_shape(self, config):
        model = Autoencoder(config)
        x = torch.randn(8, 20)
        z = model.encode(x)
        assert z.shape == (8, 4)

    def test_anomaly_score(self, config):
        model = Autoencoder(config)
        x = torch.randn(8, 20)
        scores = model.compute_anomaly_score(x)
        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_num_params(self, config):
        model = Autoencoder(config)
        assert model.get_num_params() > 0

    def test_reconstruction_improves(self, config):
        """After one gradient step, reconstruction should improve."""
        model = Autoencoder(config)
        model.train()
        x = torch.randn(16, 20)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        # Initial loss
        loss_before = torch.nn.functional.mse_loss(model(x), x).item()
        # One step
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), x)
        loss.backward()
        opt.step()
        loss_after = torch.nn.functional.mse_loss(model(x), x).item()

        assert loss_after < loss_before


class TestLSTM:
    @pytest.fixture
    def config(self):
        return {
            "model": {
                "name": "lstm",
                "input_dim": 15,
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": True,
                "use_attention": True,
            }
        }

    def test_forward_shape(self, config):
        model = LSTMAnomalyDetector(config)
        x = torch.randn(4, 10, 15)  # (batch, seq, features)
        out = model(x)
        assert out.shape == (4, 10, 15)

    def test_anomaly_score(self, config):
        model = LSTMAnomalyDetector(config)
        x = torch.randn(4, 10, 15)
        scores = model.compute_anomaly_score(x)
        assert scores.shape == (4,)

    def test_unidirectional(self):
        config = {
            "model": {
                "name": "lstm",
                "input_dim": 10,
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "use_attention": False,
            }
        }
        model = LSTMAnomalyDetector(config)
        x = torch.randn(2, 5, 10)
        out = model(x)
        assert out.shape == (2, 5, 10)


class TestTransformer:
    @pytest.fixture
    def config(self):
        return {
            "model": {
                "name": "transformer",
                "input_dim": 15,
                "d_model": 32,
                "nhead": 4,
                "num_encoder_layers": 2,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "max_seq_len": 20,
            }
        }

    def test_forward_shape(self, config):
        model = TransformerAnomalyDetector(config)
        x = torch.randn(4, 10, 15)
        out = model(x)
        assert out.shape == (4, 10, 15)

    def test_anomaly_score(self, config):
        model = TransformerAnomalyDetector(config)
        x = torch.randn(4, 10, 15)
        scores = model.compute_anomaly_score(x)
        assert scores.shape == (4,)

    def test_different_seq_lengths(self, config):
        model = TransformerAnomalyDetector(config)
        for seq_len in [5, 10, 20]:
            x = torch.randn(2, seq_len, 15)
            out = model(x)
            assert out.shape == (2, seq_len, 15)


class TestModelRegistry:
    def test_create_autoencoder(self):
        config = {
            "model": {
                "name": "autoencoder",
                "input_dim": 10,
                "encoder_dims": [8],
                "latent_dim": 4,
                "decoder_dims": [8],
                "dropout": 0.0,
                "use_batch_norm": False,
            }
        }
        model = create_model("autoencoder", config)
        assert isinstance(model, Autoencoder)

    def test_create_lstm(self):
        config = {
            "model": {
                "name": "lstm",
                "input_dim": 10,
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "use_attention": False,
            }
        }
        model = create_model("lstm", config)
        assert isinstance(model, LSTMAnomalyDetector)

    def test_create_transformer(self):
        config = {
            "model": {
                "name": "transformer",
                "input_dim": 10,
                "d_model": 16,
                "nhead": 4,
                "num_encoder_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "max_seq_len": 20,
            }
        }
        model = create_model("transformer", config)
        assert isinstance(model, TransformerAnomalyDetector)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent", {})
