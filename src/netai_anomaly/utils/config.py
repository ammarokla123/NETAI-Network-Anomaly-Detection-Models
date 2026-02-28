"""Configuration management for NETAI Anomaly Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config(
    default_path: str | Path = "configs/default.yaml",
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the default config and optionally merge a model-specific config."""
    cfg = load_config(default_path)
    if model_path is not None:
        model_cfg = load_config(model_path)
        cfg = merge_configs(cfg, model_cfg)
    return cfg


def get_device(preference: str = "auto") -> str:
    """Return the best available device string for PyTorch."""
    if preference in ("cpu", "cuda"):
        return preference
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
