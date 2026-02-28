"""LSTM-based sequence anomaly detection model.

Uses a bidirectional LSTM with optional temporal attention to reconstruct
network telemetry sequences.  High reconstruction error flags anomalies.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from netai_anomaly.models.base import BaseAnomalyModel, register_model


class TemporalAttention(nn.Module):
    """Simple additive attention over time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        # lstm_output: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(lstm_output), dim=1)  # (B, T, 1)
        context = (lstm_output * weights).sum(dim=1)  # (B, H)
        return context, weights.squeeze(-1)


@register_model("lstm")
class LSTMAnomalyDetector(BaseAnomalyModel):
    """Bidirectional LSTM encoder-decoder for sequence anomaly detection."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        mcfg = config["model"]
        self.input_dim: int = mcfg["input_dim"]
        self.hidden_dim: int = mcfg.get("hidden_dim", 64)
        self.num_layers: int = mcfg.get("num_layers", 2)
        dropout: float = mcfg.get("dropout", 0.2)
        self.bidirectional: bool = mcfg.get("bidirectional", True)
        self.use_attention: bool = mcfg.get("use_attention", True)

        self.num_directions = 2 if self.bidirectional else 1
        lstm_hidden = self.hidden_dim

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        enc_output_dim = lstm_hidden * self.num_directions

        # Optional attention
        self.attention = TemporalAttention(enc_output_dim) if self.use_attention else None

        # Decoder: project back to input space at each time step
        self.decoder = nn.Sequential(
            nn.Linear(enc_output_dim, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        enc_out, _ = self.encoder_lstm(x)  # (B, T, H*D)

        # Reconstruct each time step
        reconstructed = self.decoder(enc_out)  # (B, T, input_dim)
        return reconstructed
