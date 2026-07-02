"""
src/deep/trainer.py

Training loop for the deep learning pipeline.

Requirements: 10.3–10.9, 16.2
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.exceptions import ArtifactError, ConfigError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config_loader import Config

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    """Records metrics from a completed training run."""
    epoch_val_accuracies: list[float] = field(default_factory=list)
    best_epoch: int = 0
    total_train_time_seconds: float = 0.0


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace).

    Args:
        in_features: Embedding dimension.
        num_classes: Number of identity classes.
        margin: Angular margin m (default 0.5).
        scale: Scale factor s (default 64).
    """

    def __init__(self, in_features: int, num_classes: int, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(embeddings),
            torch.nn.functional.normalize(self.weight),
        )
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_margin - sine * self.sin_margin
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        return nn.functional.cross_entropy(logits, labels)


class Trainer:
    """Trains a PyTorch model with the configured optimizer and loss function."""

    def __init__(self, model: nn.Module, config: "Config") -> None:
        self._model = model
        self._config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Raise ConfigError before training if any required param is invalid (R10.8)."""
        cfg = self._config
        errors = []
        if not isinstance(cfg.epochs, int) or cfg.epochs < 1:
            errors.append(f"epochs must be a positive integer, got '{cfg.epochs}'")
        if not isinstance(cfg.learning_rate, (int, float)) or cfg.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got '{cfg.learning_rate}'")
        if not isinstance(cfg.batch_size, int) or cfg.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got '{cfg.batch_size}'")
        if cfg.optimizer.lower() not in {"adam", "sgd"}:
            errors.append(f"optimizer must be 'adam' or 'sgd', got '{cfg.optimizer}'")
        if errors:
            raise ConfigError("Invalid training config: " + "; ".join(errors))

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Run the training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.

        Returns:
            TrainingHistory with per-epoch validation accuracies and timing.
        """
        cfg = self._config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on device: %s", device)
        self._model.to(device)

        # Build optimizer (R10.3)
        if cfg.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg.learning_rate)
        else:
            optimizer = torch.optim.SGD(
                self._model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=1e-4
            )

        # Loss function (R10.7)
        if cfg.arcface_enabled:
            # Determine embedding size from model
            num_classes = self._get_num_classes()
            emb_dim = self._get_embedding_dim(train_loader, device)
            criterion = ArcFaceLoss(emb_dim, num_classes, cfg.arcface_margin, cfg.arcface_scale).to(device)
        else:
            criterion = nn.CrossEntropyLoss()

        history = TrainingHistory()
        best_val_acc = -1.0
        start_time = time.time()

        try:
            for epoch in range(1, cfg.epochs + 1):
                self._model.train()
                total_loss = 0.0
                for batch_imgs, batch_labels in train_loader:
                    batch_imgs = batch_imgs.to(device)
                    batch_labels = batch_labels.to(device)

                    optimizer.zero_grad()
                    outputs = self._model(batch_imgs)

                    if cfg.arcface_enabled:
                        loss = criterion(outputs, batch_labels)
                    else:
                        loss = criterion(outputs, batch_labels)

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Validation after each epoch (R10.4)
                val_acc = self._evaluate_top1(self._model, val_loader, device)
                history.epoch_val_accuracies.append(val_acc)

                avg_loss = total_loss / max(len(train_loader), 1)
                logger.info(
                    "Epoch %d/%d — loss=%.4f  val_top1=%.4f",
                    epoch, cfg.epochs, avg_loss, val_acc,
                )

                # Save checkpoint if improved (R10.5, R16.2)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    history.best_epoch = epoch
                    self._save_checkpoint(optimizer, epoch, val_acc)

        except KeyboardInterrupt:
            # R10.9: Preserve last saved checkpoint on interrupt
            logger.warning("Training interrupted by user. Last saved checkpoint is preserved.")

        history.total_train_time_seconds = time.time() - start_time
        logger.info(
            "Training complete: best val_top1=%.4f at epoch %d (%.1fs total)",
            best_val_acc, history.best_epoch, history.total_train_time_seconds,
        )
        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_top1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
        """Compute top-1 validation accuracy."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def _save_checkpoint(self, optimizer: torch.optim.Optimizer, epoch: int, val_acc: float) -> None:
        """Save the current model checkpoint to disk."""
        path = Path(self._config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        }
        try:
            torch.save(checkpoint, path)
            logger.info("Checkpoint saved (epoch=%d, val_acc=%.4f) → %s", epoch, val_acc, path)
        except Exception as exc:
            raise ArtifactError(f"Failed to save checkpoint to '{path}': {exc}") from exc

    def _get_num_classes(self) -> int:
        """Infer number of output classes from the model's final layer."""
        for name in ["fc", "classifier"]:
            layer = getattr(self._model, name, None)
            if layer is not None:
                if isinstance(layer, nn.Linear):
                    return layer.out_features
                if isinstance(layer, nn.Sequential):
                    for sub in reversed(list(layer.children())):
                        if isinstance(sub, nn.Linear):
                            return sub.out_features
        return 1

    def _get_embedding_dim(self, loader: DataLoader, device: torch.device) -> int:
        """Get the embedding dimension by running a single forward pass."""
        self._model.eval()
        with torch.no_grad():
            for imgs, _ in loader:
                out = self._model(imgs.to(device))
                return out.shape[1]
        return 512
