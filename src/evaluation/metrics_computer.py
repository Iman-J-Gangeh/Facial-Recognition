"""
src/evaluation/metrics_computer.py

Computes evaluation metrics for both pipelines from EvaluationOutput.

Requirements: 12.1–12.6
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

from src.utils.artifact_manager import ArtifactManager
from src.utils.logger import get_logger

logger = get_logger(__name__)
_artifact_manager = ArtifactManager()


@dataclass
class MetricsResult:
    """All computed metrics for one pipeline."""
    pipeline: str
    top1: float
    top5: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: np.ndarray
    train_time_s: float          # total wall-clock training time in seconds
    mean_inference_ms: float     # mean per-image inference time in milliseconds
    artifact_size_mb: float      # total size of all saved artifacts in MB
    roc_data: dict[str, Any] | None = None


class MetricsComputer:
    """Computes and records evaluation metrics."""

    def __init__(self, config: Any = None) -> None:
        self._config = config

    def compute(
        self,
        y_true: list[str],
        y_pred: list[str],
        scores: np.ndarray,
        pipeline_name: str,
        train_time_s: float,
        mean_inference_ms: float,
        artifact_paths: list[str] | None = None,
    ) -> MetricsResult:
        """Compute all required metrics.

        Args:
            y_true: Ground-truth identity labels.
            y_pred: Predicted identity labels.
            scores: (N, num_classes) probability/score matrix.
            pipeline_name: "classical" or "deep".
            train_time_s: Training time in seconds.
            mean_inference_ms: Mean per-image inference time in ms.
            artifact_paths: List of artifact file paths for size measurement.

        Returns:
            MetricsResult with all computed metrics.
        """
        # Get unique classes from y_true (ignoring any 'unknown' predictions)
        classes = sorted(set(y_true))

        # Replace any 'unknown' predictions with the most frequent true class
        y_pred_clean = [
            p if p in classes else (classes[0] if classes else p)
            for p in y_pred
        ]

        # Top-1 accuracy
        top1 = sum(t == p for t, p in zip(y_true, y_pred_clean)) / max(len(y_true), 1)

        # Top-5 accuracy
        top5 = self._compute_top5(y_true, scores, classes)

        # Macro precision, recall, F1
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_clean, average="macro", zero_division=0, labels=classes
        )

        # Per-class accuracy
        per_class = self._per_class_accuracy(y_true, y_pred_clean, classes)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_clean, labels=classes)

        # Artifact size
        size_mb = _artifact_manager.measure_size_mb(artifact_paths or [])

        # Optional ROC/AUC/FAR/FRR/EER
        roc_data = None
        if self._config is not None and getattr(self._config, "roc_enabled", False):
            roc_data = self._compute_roc(y_true, scores, classes)

        return MetricsResult(
            pipeline=pipeline_name,
            top1=float(top1),
            top5=float(top5),
            macro_precision=float(prec),
            macro_recall=float(rec),
            macro_f1=float(f1),
            per_class_accuracy=per_class,
            confusion_matrix=cm,
            train_time_s=float(train_time_s),
            mean_inference_ms=float(mean_inference_ms),
            artifact_size_mb=float(size_mb),
            roc_data=roc_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_top5(
        y_true: list[str],
        scores: np.ndarray,
        classes: list[str],
    ) -> float:
        """Compute top-5 accuracy using the scores matrix."""
        if scores is None or scores.size == 0 or len(classes) < 2:
            return 0.0

        label_to_idx = {c: i for i, c in enumerate(classes)}
        y_true_idx = [label_to_idx.get(t, 0) for t in y_true]

        # Align scores columns to classes
        if scores.shape[1] != len(classes):
            # Pad or truncate scores to match classes
            aligned = np.zeros((len(y_true), len(classes)), dtype=np.float32)
            n_cols = min(scores.shape[1], len(classes))
            aligned[:, :n_cols] = scores[:, :n_cols]
            scores = aligned

        k = min(5, len(classes))
        try:
            return float(top_k_accuracy_score(y_true_idx, scores, k=k, labels=list(range(len(classes)))))
        except Exception:
            return 0.0

    @staticmethod
    def _per_class_accuracy(
        y_true: list[str],
        y_pred: list[str],
        classes: list[str],
    ) -> dict[str, float]:
        """Compute per-class accuracy."""
        per_class: dict[str, float] = {}
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        for cls in classes:
            mask = y_true_arr == cls
            if mask.sum() == 0:
                per_class[cls] = 0.0
            else:
                per_class[cls] = float((y_pred_arr[mask] == cls).sum() / mask.sum())
        return per_class

    @staticmethod
    def _compute_roc(
        y_true: list[str],
        scores: np.ndarray,
        classes: list[str],
    ) -> dict[str, Any]:
        """Compute per-class ROC, AUC, FAR, FRR, EER by threshold sweep."""
        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.preprocessing import label_binarize

        if scores is None or scores.size == 0:
            return {}

        y_bin = label_binarize(y_true, classes=classes)
        roc_results: dict[str, Any] = {}

        for i, cls in enumerate(classes):
            if y_bin.shape[1] <= i:
                continue
            try:
                fpr, tpr, thresholds = roc_curve(y_bin[:, i], scores[:, i])
                auc_val = float(roc_auc_score(y_bin[:, i], scores[:, i]))

                # FAR = FPR, FRR = 1 - TPR
                far = fpr
                frr = 1.0 - tpr
                # EER: point where FAR ≈ FRR
                eer_idx = np.argmin(np.abs(far - frr))
                eer = float((far[eer_idx] + frr[eer_idx]) / 2.0)

                roc_results[cls] = {
                    "auc": auc_val,
                    "eer": eer,
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "far": far.tolist(),
                    "frr": frr.tolist(),
                }
            except Exception as exc:
                logger.warning("ROC computation failed for class '%s': %s", cls, exc)

        return roc_results
