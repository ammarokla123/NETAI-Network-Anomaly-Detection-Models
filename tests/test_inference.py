"""Tests for the FastAPI inference service."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from netai_anomaly.inference.service import app, _state
from netai_anomaly.models.autoencoder import Autoencoder


@pytest.fixture
def client():
    """Create a test client with a loaded model."""
    config = {
        "model": {
            "name": "autoencoder",
            "input_dim": 5,
            "encoder_dims": [8],
            "latent_dim": 4,
            "decoder_dims": [8],
            "dropout": 0.0,
            "use_batch_norm": False,
        },
        "data": {"sequence_length": 10},
    }
    model = Autoencoder(config)
    model.eval()

    _state["model"] = model
    _state["config"] = config
    _state["device"] = "cpu"
    _state["threshold"] = 0.5
    _state["scaler_mean"] = None
    _state["scaler_scale"] = None

    with TestClient(app) as c:
        yield c

    # Cleanup
    _state["model"] = None


@pytest.fixture
def client_no_model():
    """Test client without a loaded model."""
    _state["model"] = None
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_with_model(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "autoencoder"

    def test_health_no_model(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_model"


class TestPredictEndpoint:
    def test_predict_single(self, client):
        resp = client.post("/predict", json={
            "throughput_mbps": 9500.0,
            "latency_ms": 5.0,
            "packet_loss_pct": 0.01,
            "retransmits": 2,
            "jitter_ms": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert "threshold" in data
        assert "confidence" in data
        assert isinstance(data["is_anomaly"], bool)

    def test_predict_no_model(self, client_no_model):
        resp = client_no_model.post("/predict", json={
            "throughput_mbps": 100.0,
            "latency_ms": 5.0,
            "packet_loss_pct": 0.0,
            "retransmits": 0,
            "jitter_ms": 0.1,
        })
        assert resp.status_code == 503

    def test_predict_invalid_input(self, client):
        resp = client.post("/predict", json={"throughput_mbps": 100.0})
        assert resp.status_code == 422  # validation error


class TestBatchEndpoint:
    def test_batch_predict(self, client):
        samples = [
            {
                "throughput_mbps": 9500.0,
                "latency_ms": 5.0,
                "packet_loss_pct": 0.01,
                "retransmits": 2,
                "jitter_ms": 0.5,
            },
            {
                "throughput_mbps": 100.0,
                "latency_ms": 500.0,
                "packet_loss_pct": 25.0,
                "retransmits": 200,
                "jitter_ms": 50.0,
            },
        ]
        resp = client.post("/predict/batch", json={"samples": samples})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        assert "summary" in data
        assert data["summary"]["total"] == 2

    def test_batch_empty(self, client):
        resp = client.post("/predict/batch", json={"samples": []})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 0


class TestModelInfoEndpoint:
    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "num_parameters" in data
        assert "threshold" in data

    def test_model_info_no_model(self, client_no_model):
        resp = client_no_model.get("/model/info")
        assert resp.status_code == 503
