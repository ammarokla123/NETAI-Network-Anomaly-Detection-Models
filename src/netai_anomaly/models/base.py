"""Base model class and model registry for anomaly detection models."""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn


class BaseAnomalyModel(nn.Module, abc.ABC):
    """Abstract base class for all anomaly detection models.

    All models operate as reconstruction-based detectors: they learn to
    reconstruct *normal* data, and anomalies are flagged when the
    reconstruction error exceeds a learned threshold.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the reconstruction."""
        ...

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error (MSE)."""
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(x)
            # Mean squared error per sample
            if x_hat.dim() == 3 and x.dim() == 3:
                score = ((x - x_hat) ** 2).mean(dim=(1, 2))
            else:
                score = ((x - x_hat) ** 2).mean(dim=-1)
        return score

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, type[BaseAnomalyModel]] = {}


def register_model(name: str):
    """Decorator to register a model class under a name."""

    def decorator(cls: type[BaseAnomalyModel]):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def create_model(name: str, config: dict[str, Any]) -> BaseAnomalyModel:
    """Instantiate a registered model by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](config)
