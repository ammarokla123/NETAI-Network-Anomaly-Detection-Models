"""FastAPI real-time inference service for network anomaly detection.

Exposes REST endpoints for single-sample and batch anomaly detection,
health checks, and model metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from netai_anomaly.models.base import BaseAnomalyModel, create_model

# Ensure model classes are registered
import netai_anomaly.models.autoencoder  # noqa: F401
import netai_anomaly.models.lstm  # noqa: F401
import netai_anomaly.models.transformer  # noqa: F401

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NETAI Anomaly Detection Service",
    description="Real-time network anomaly detection using deep learning models",
    version="0.1.0",
)

# Global state (populated by load_model)
_state: dict[str, Any] = {
    "model": None,
    "threshold": 0.0,
    "config": None,
    "device": "cpu",
    "scaler_mean": None,
    "scaler_scale": None,
}

FEATURE_NAMES = [
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "retransmits",
    "jitter_ms",
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TelemetrySample(BaseModel):
    """A single network telemetry measurement."""

    throughput_mbps: float = Field(..., description="Throughput in Mbps")
    latency_ms: float = Field(..., description="Latency in milliseconds")
    packet_loss_pct: float = Field(..., ge=0, le=100, description="Packet loss percentage")
    retransmits: int = Field(..., ge=0, description="Number of retransmits")
    jitter_ms: float = Field(..., ge=0, description="Jitter in milliseconds")


class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    confidence: float = Field(..., description="Confidence score (0-1)")


class BatchRequest(BaseModel):
    samples: list[TelemetrySample]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    num_parameters: int | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _to_tensor(sample: TelemetrySample) -> torch.Tensor:
    """Convert a sample to a normalized tensor."""
    arr = np.array(
        [
            [
                sample.throughput_mbps,
                sample.latency_ms,
                sample.packet_loss_pct,
                sample.retransmits,
                sample.jitter_ms,
            ]
        ],
        dtype=np.float32,
    )
    # Apply saved normalization if available
    if _state["scaler_mean"] is not None:
        arr = (arr - _state["scaler_mean"]) / (_state["scaler_scale"] + 1e-8)
    return torch.tensor(arr, dtype=torch.float32).to(_state["device"])


def _predict_single(sample: TelemetrySample) -> PredictionResponse:
    model: BaseAnomalyModel = _state["model"]
    x = _to_tensor(sample)

    # For sequence models, repeat to fill sequence length
    if hasattr(model, "encoder_lstm") or hasattr(model, "transformer_encoder"):
        seq_len = _state["config"].get("data", {}).get("sequence_length", 60)
        x = x.unsqueeze(0).expand(-1, seq_len, -1)

    score = float(model.compute_anomaly_score(x).item())
    threshold = _state["threshold"]
    is_anomaly = score >= threshold

    # Confidence: how far score is from threshold, normalized
    if threshold > 0:
        confidence = min(1.0, abs(score - threshold) / threshold)
    else:
        confidence = 1.0

    return PredictionResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 6),
        threshold=round(threshold, 6),
        confidence=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    model = _state["model"]
    return HealthResponse(
        status="healthy" if model is not None else "no_model",
        model_loaded=model is not None,
        model_name=_state["config"]["model"]["name"] if _state["config"] else None,
        num_parameters=model.get_num_params() if model else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(sample: TelemetrySample):
    """Predict whether a single telemetry sample is anomalous."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return _predict_single(sample)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Batch prediction for multiple telemetry samples."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    predictions = [_predict_single(s) for s in request.samples]
    num_anomalies = sum(1 for p in predictions if p.is_anomaly)

    return BatchResponse(
        predictions=predictions,
        summary={
            "total": len(predictions),
            "anomalies": num_anomalies,
            "normal": len(predictions) - num_anomalies,
            "anomaly_rate": round(num_anomalies / max(len(predictions), 1), 4),
        },
    )


@app.get("/model/info")
def model_info():
    """Return model configuration and metadata."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return {
        "config": _state["config"],
        "num_parameters": _state["model"].get_num_params(),
        "threshold": _state["threshold"],
        "device": _state["device"],
    }


# ---------------------------------------------------------------------------
# Model loading (called at startup or externally)
# ---------------------------------------------------------------------------
def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    threshold: float | None = None,
) -> None:
    """Load a trained model from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = create_model(config["model"]["name"], config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _state["model"] = model
    _state["config"] = config
    _state["device"] = device
    _state["threshold"] = threshold if threshold is not None else 0.01

    logger.info("Loaded model '%s' from %s", config["model"]["name"], checkpoint_path)
