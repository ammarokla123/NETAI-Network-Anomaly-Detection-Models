"""PyTorch Dataset classes for network telemetry sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from netai_anomaly.data.features import FeaturePipeline


class TelemetryDataset(Dataset):
    """Sliding-window dataset over network telemetry features.

    Each sample is a ``(sequence, label)`` pair where *sequence* has shape
    ``(seq_len, num_features)`` and *label* is 1 if **any** time-step in the
    window is anomalous.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        seq_len: int = 60,
    ):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
        self.n_samples = len(features) - seq_len + 1

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = idx + self.seq_len
        seq = torch.tensor(self.features[idx:end], dtype=torch.float32)

        if self.labels is not None:
            # Window is anomalous if any step in the window is anomalous
            label = torch.tensor(
                float(self.labels[idx:end].max()), dtype=torch.float32
            )
        else:
            label = torch.tensor(0.0, dtype=torch.float32)

        return seq, label


class FlatDataset(Dataset):
    """Non-sequential (flat) dataset for the Autoencoder.

    Each sample is a single feature vector (not a sequence).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray | None = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32)
            if labels is not None
            else torch.zeros(len(features))
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def build_datasets(
    df: pd.DataFrame,
    pipeline: FeaturePipeline,
    seq_len: int = 60,
    train_split: float = 0.7,
    val_split: float = 0.15,
    flat: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    """Build train / val / test datasets from a telemetry DataFrame.

    Parameters
    ----------
    flat : bool
        If True, return ``FlatDataset`` instances (for Autoencoder).
        If False, return ``TelemetryDataset`` with sliding windows.
    """
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    X_train, y_train = pipeline.fit_transform(train_df)
    X_val, y_val = pipeline.transform(val_df)
    X_test, y_test = pipeline.transform(test_df)

    DatasetClass = FlatDataset if flat else TelemetryDataset
    kwargs = {} if flat else {"seq_len": seq_len}

    return (
        DatasetClass(X_train, y_train, **kwargs),
        DatasetClass(X_val, y_val, **kwargs),
        DatasetClass(X_test, y_test, **kwargs),
    )
