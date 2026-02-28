"""Training loop and utilities for anomaly detection models."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from netai_anomaly.models.base import BaseAnomalyModel

logger = logging.getLogger(__name__)


def _get_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]):
    name = cfg.get("scheduler", "cosine")
    epochs = cfg.get("epochs", 50)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    return None


class Trainer:
    """Generic trainer for reconstruction-based anomaly detection models."""

    def __init__(
        self,
        model: BaseAnomalyModel,
        config: dict[str, Any],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        tcfg = config.get("training", {})

        self.epochs: int = tcfg.get("epochs", 50)
        self.lr: float = tcfg.get("learning_rate", 0.001)
        self.weight_decay: float = tcfg.get("weight_decay", 1e-4)
        self.patience: int = tcfg.get("patience", 10)
        self.grad_clip: float = tcfg.get("gradient_clip", 1.0)
        self.checkpoint_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = _get_scheduler(self.optimizer, tcfg)

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float("inf")
        self._patience_counter = 0

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            x_hat = self.model(batch_x)
            loss = self.criterion(x_hat, batch_x)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device)
            x_hat = self.model(batch_x)
            loss = self.criterion(x_hat, batch_x)
            total_loss += loss.item() * batch_x.size(0)
        return total_loss / len(loader.dataset)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Train the model and return training history."""
        model_name = self.config.get("model", {}).get("name", "model")
        logger.info(
            "Training %s | %d params | device=%s",
            model_name,
            self.model.get_num_params(),
            self.device,
        )

        start = time.time()
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # LR scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                self.optimizer.param_groups[0]["lr"],
            )

            # Checkpointing & early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._patience_counter = 0
                self._save_checkpoint(f"{model_name}_best.pt")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        elapsed = time.time() - start
        logger.info("Training complete in %.1fs. Best val_loss=%.6f", elapsed, self.best_val_loss)

        # Load best model
        self._load_checkpoint(f"{model_name}_best.pt")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.train_losses),
            "elapsed_seconds": elapsed,
        }

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def _load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

    @torch.no_grad()
    def compute_threshold(self, loader: DataLoader, percentile: float = 95.0) -> float:
        """Compute the anomaly threshold from reconstruction errors on normal data."""
        self.model.eval()
        scores: list[float] = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device)
            s = self.model.compute_anomaly_score(batch_x)
            scores.extend(s.cpu().numpy().tolist())
        threshold = float(np.percentile(scores, percentile))
        logger.info("Anomaly threshold (p%.0f): %.6f", percentile, threshold)
        return threshold
