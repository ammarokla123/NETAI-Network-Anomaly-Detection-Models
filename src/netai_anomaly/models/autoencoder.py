"""Autoencoder model for unsupervised network anomaly detection.

The autoencoder learns a compressed latent representation of normal network
behaviour.  High reconstruction error on new data indicates an anomaly.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from netai_anomaly.models.base import BaseAnomalyModel, register_model


@register_model("autoencoder")
class Autoencoder(BaseAnomalyModel):
    """Fully-connected autoencoder with configurable encoder/decoder layers."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        mcfg = config["model"]
        input_dim: int = mcfg["input_dim"]
        encoder_dims: list[int] = mcfg.get("encoder_dims", [64, 32, 16])
        latent_dim: int = mcfg.get("latent_dim", 8)
        decoder_dims: list[int] = mcfg.get("decoder_dims", [16, 32, 64])
        dropout: float = mcfg.get("dropout", 0.2)
        use_bn: bool = mcfg.get("use_batch_norm", True)

        # Build encoder
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for dim in encoder_dims:
            enc_layers.append(nn.Linear(prev, dim))
            if use_bn:
                enc_layers.append(nn.BatchNorm1d(dim))
            enc_layers.append(nn.ReLU(inplace=True))
            enc_layers.append(nn.Dropout(dropout))
            prev = dim
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder
        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for dim in decoder_dims:
            dec_layers.append(nn.Linear(prev, dim))
            if use_bn:
                dec_layers.append(nn.BatchNorm1d(dim))
            dec_layers.append(nn.ReLU(inplace=True))
            dec_layers.append(nn.Dropout(dropout))
            prev = dim
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x)
