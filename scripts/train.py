#!/usr/bin/env python3
"""Train an anomaly detection model on network telemetry data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from netai_anomaly.data.dataset import build_datasets
from netai_anomaly.data.features import FeaturePipeline
from netai_anomaly.data.generator import load_from_sqlite
from netai_anomaly.models.base import create_model
from netai_anomaly.training.trainer import Trainer
from netai_anomaly.training.utils import set_seed
from netai_anomaly.utils.config import get_config, get_device

# Ensure model classes are registered
import netai_anomaly.models.autoencoder  # noqa: F401
import netai_anomaly.models.lstm  # noqa: F401
import netai_anomaly.models.transformer  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument(
        "--model",
        choices=["autoencoder", "lstm", "transformer"],
        required=True,
        help="Model architecture",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Default config path")
    parser.add_argument("--db-path", default=None, help="Override DB path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/auto)")
    args = parser.parse_args()

    # Load and merge configs
    model_config_path = f"configs/{args.model}.yaml"
    cfg = get_config(args.config, model_config_path)

    # Apply CLI overrides
    if args.db_path:
        cfg["data"]["db_path"] = args.db_path
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    device = get_device(args.device or cfg["training"].get("device", "auto"))
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    logger.info("Configuration: model=%s, device=%s", args.model, device)

    # Load data
    db_path = cfg["data"]["db_path"]
    logger.info("Loading data from %s", db_path)
    df = load_from_sqlite(db_path)
    logger.info("Loaded %d samples (%d anomalies)", len(df), df["is_anomaly"].sum())

    # Build datasets
    pipeline = FeaturePipeline(cfg)
    seq_len = cfg["data"].get("sequence_length", 60)
    is_flat = args.model == "autoencoder"

    train_ds, val_ds, test_ds = build_datasets(
        df,
        pipeline,
        seq_len=seq_len,
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        flat=is_flat,
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Set dynamic input_dim
    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]
    cfg["model"]["input_dim"] = input_dim
    if cfg["model"].get("output_dim") is None:
        cfg["model"]["output_dim"] = input_dim

    logger.info("Feature dimension: %d | Train: %d | Val: %d | Test: %d",
                input_dim, len(train_ds), len(val_ds), len(test_ds))

    # Create model
    model = create_model(args.model, cfg)
    logger.info("Model: %s (%d parameters)", args.model, model.get_num_params())

    # Train
    trainer = Trainer(model, cfg, device=device)
    history = trainer.fit(train_loader, val_loader)

    # Compute threshold on validation set
    threshold = trainer.compute_threshold(
        val_loader, percentile=cfg["evaluation"]["threshold_percentile"]
    )

    # Save training results
    output_dir = Path("outputs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "num_parameters": model.get_num_params(),
        "input_dim": input_dim,
        "threshold": threshold,
        "history": {
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "best_val_loss": history["best_val_loss"],
            "epochs_trained": history["epochs_trained"],
        },
    }

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_dir)
    logger.info("Best val loss: %.6f | Threshold: %.6f", history["best_val_loss"], threshold)


if __name__ == "__main__":
    main()
