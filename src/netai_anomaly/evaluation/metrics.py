"""Evaluation metrics for anomaly detection models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from netai_anomaly.models.base import BaseAnomalyModel


def compute_scores(
    model: BaseAnomalyModel,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute anomaly scores and collect true labels from a data loader.

    Returns
    -------
    scores : np.ndarray of shape (N,)
    labels : np.ndarray of shape (N,)
    """
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            s = model.compute_anomaly_score(batch_x)
            all_scores.append(s.cpu().numpy())
            all_labels.append(batch_y.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Parameters
    ----------
    scores : per-sample anomaly scores (higher = more anomalous)
    labels : ground-truth binary labels (1 = anomaly)
    threshold : decision threshold on scores
    """
    preds = (scores >= threshold).astype(int)
    labels_int = labels.astype(int)

    results: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels_int, preds)),
        "precision": float(precision_score(labels_int, preds, zero_division=0)),
        "recall": float(recall_score(labels_int, preds, zero_division=0)),
        "f1_score": float(f1_score(labels_int, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels_int, preds).tolist(),
        "classification_report": classification_report(
            labels_int, preds, target_names=["Normal", "Anomaly"], zero_division=0
        ),
    }

    # ROC & PR curves (require both classes present)
    if len(np.unique(labels_int)) > 1:
        results["roc_auc"] = float(roc_auc_score(labels_int, scores))
        results["pr_auc"] = float(average_precision_score(labels_int, scores))
        fpr, tpr, _ = roc_curve(labels_int, scores)
        results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        prec_arr, rec_arr, _ = precision_recall_curve(labels_int, scores)
        results["pr_curve"] = {"precision": prec_arr.tolist(), "recall": rec_arr.tolist()}
    else:
        results["roc_auc"] = None
        results["pr_auc"] = None

    return results
