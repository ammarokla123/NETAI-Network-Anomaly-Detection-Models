#!/usr/bin/env python3
"""Evaluate a trained anomaly detection model and generate reports."""

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
from netai_anomaly.evaluation.metrics import compute_scores, evaluate
from netai_anomaly.evaluation.visualize import save_all_plots
from netai_anomaly.models.base import create_model
from netai_anomaly.utils.config import get_device

import netai_anomaly.models.autoencoder  # noqa: F401
import netai_anomaly.models.lstm  # noqa: F401
import netai_anomaly.models.transformer  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--db-path", default="data/network_telemetry.db", help="SQLite DB path")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")
    args = parser.parse_args()

    device = get_device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_name = cfg["model"]["name"]

    output_dir = Path(args.output_dir or f"outputs/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recreate model
    model = create_model(model_name, cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Loaded %s (%d params) from %s", model_name, model.get_num_params(), args.checkpoint)

    # Load and prepare data
    df = load_from_sqlite(args.db_path)
    pipeline = FeaturePipeline(cfg)
    seq_len = cfg["data"].get("sequence_length", 60)
    is_flat = model_name == "autoencoder"

    _, _, test_ds = build_datasets(
        df, pipeline, seq_len=seq_len,
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        flat=is_flat,
    )

    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # Compute scores
    scores, labels = compute_scores(model, test_loader, device=device)

    # Load threshold
    training_results_path = output_dir / "training_results.json"
    if training_results_path.exists():
        with open(training_results_path) as f:
            tr = json.load(f)
        threshold = tr["threshold"]
    else:
        import numpy as np
        threshold = float(np.percentile(scores, cfg["evaluation"]["threshold_percentile"]))

    # Evaluate
    results = evaluate(scores, labels, threshold)

    logger.info("=== Evaluation Results ===")
    logger.info("Accuracy:  %.4f", results["accuracy"])
    logger.info("Precision: %.4f", results["precision"])
    logger.info("Recall:    %.4f", results["recall"])
    logger.info("F1 Score:  %.4f", results["f1_score"])
    if results["roc_auc"] is not None:
        logger.info("ROC AUC:   %.4f", results["roc_auc"])
        logger.info("PR AUC:    %.4f", results["pr_auc"])
    logger.info("\n%s", results["classification_report"])

    # Save metrics
    serializable = {k: v for k, v in results.items() if k != "classification_report"}
    serializable["classification_report_text"] = results["classification_report"]
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Generate plots
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])
    try:
        save_all_plots(results, train_losses, val_losses, scores, labels, output_dir)
        logger.info("Plots saved to %s", output_dir)
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots")

    logger.info("Evaluation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
