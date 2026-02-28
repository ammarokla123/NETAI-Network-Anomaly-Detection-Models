"""Visualisation helpers for evaluation results.

Generates matplotlib figures for training curves, ROC, PR, and score
distributions.  Designed to be imported by scripts or notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Guard matplotlib import so the library works without the `plots` extra
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("Install matplotlib & seaborn: pip install netai-anomaly[plots]")


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str | Path | None = None,
) -> None:
    """Plot training and validation loss curves."""
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    roc_data: dict[str, list[float]],
    auc: float | None,
    save_path: str | Path | None = None,
) -> None:
    """Plot ROC curve."""
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc_data["fpr"], roc_data["tpr"], label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(
    pr_data: dict[str, list[float]],
    auc: float | None,
    save_path: str | Path | None = None,
) -> None:
    """Plot Precision-Recall curve."""
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pr_data["recall"], pr_data["precision"], label=f"PR (AUC = {auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_path: str | Path | None = None,
) -> None:
    """Plot anomaly score distributions for normal and anomalous samples."""
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    ax.hist(normal_scores, bins=80, alpha=0.6, label="Normal", density=True, color="steelblue")
    if len(anomaly_scores) > 0:
        ax.hist(
            anomaly_scores, bins=80, alpha=0.6, label="Anomaly", density=True, color="crimson"
        )
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_all_plots(
    results: dict[str, Any],
    train_losses: list[float],
    val_losses: list[float],
    scores: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path = "outputs",
) -> None:
    """Generate and save all evaluation plots."""
    _check_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(train_losses, val_losses, output_dir / "training_curves.png")

    if results.get("roc_curve"):
        plot_roc_curve(results["roc_curve"], results.get("roc_auc"), output_dir / "roc_curve.png")

    if results.get("pr_curve"):
        plot_pr_curve(results["pr_curve"], results.get("pr_auc"), output_dir / "pr_curve.png")

    plot_score_distribution(
        scores, labels, results["threshold"], output_dir / "score_distribution.png"
    )
