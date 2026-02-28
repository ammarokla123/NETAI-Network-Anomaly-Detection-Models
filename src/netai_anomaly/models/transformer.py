"""Transformer-based anomaly detection model.

Uses a Transformer encoder with positional encoding to model temporal
dependencies in network telemetry, followed by a reconstruction head.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from netai_anomaly.models.base import BaseAnomalyModel, register_model


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@register_model("transformer")
class TransformerAnomalyDetector(BaseAnomalyModel):
    """Transformer encoder with reconstruction decoder for anomaly detection."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        mcfg = config["model"]
        self.input_dim: int = mcfg["input_dim"]
        d_model: int = mcfg.get("d_model", 64)
        nhead: int = mcfg.get("nhead", 4)
        num_layers: int = mcfg.get("num_encoder_layers", 3)
        dim_ff: int = mcfg.get("dim_feedforward", 128)
        dropout: float = mcfg.get("dropout", 0.1)
        max_seq_len: int = mcfg.get("max_seq_len", 60)

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Reconstruction head
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)         # (B, T, d_model)
        x = self.pos_encoder(x)        # (B, T, d_model)
        x = self.transformer_encoder(x)  # (B, T, d_model)
        x = self.output_proj(x)        # (B, T, input_dim)
        return x
