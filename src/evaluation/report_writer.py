"""
src/evaluation/report_writer.py

Saves MetricsResult to JSON and CSV files.

Requirements: 12.5
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics_computer import MetricsResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_serializable(obj: Any) -> Any:
    """Convert numpy types and arrays to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class ReportWriter:
    """Writes MetricsResult to outputs/results/ as JSON and CSV."""

    def __init__(self, results_dir: str = "outputs/results") -> None:
        self._results_dir = Path(results_dir)

    def save(self, result: MetricsResult) -> None:
        """Persist a MetricsResult to JSON and CSV.

        Files are named <pipeline>_metrics.json and <pipeline>_metrics.csv.

        Args:
            result: The MetricsResult to persist.
        """
        self._results_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "pipeline": result.pipeline,
            "top1": result.top1,
            "top5": result.top5,
            "macro_precision": result.macro_precision,
            "macro_recall": result.macro_recall,
            "macro_f1": result.macro_f1,
            "per_class_accuracy": result.per_class_accuracy,
            "confusion_matrix": result.confusion_matrix.tolist() if result.confusion_matrix is not None else [],
            "train_time_s": result.train_time_s,
            "mean_inference_ms": result.mean_inference_ms,
            "artifact_size_mb": result.artifact_size_mb,
            "roc_data": result.roc_data,
        }

        # JSON
        json_path = self._results_dir / f"{result.pipeline}_metrics.json"
        try:
            with json_path.open("w", encoding="utf-8") as fh:
                json.dump(_to_serializable(data), fh, indent=2)
            logger.info("Metrics JSON saved → %s", json_path)
        except Exception as exc:
            logger.error("Failed to write JSON metrics for '%s': %s", result.pipeline, exc)

        # CSV — flat scalar metrics only
        csv_path = self._results_dir / f"{result.pipeline}_metrics.csv"
        try:
            flat = {
                "pipeline": result.pipeline,
                "top1": result.top1,
                "top5": result.top5,
                "macro_precision": result.macro_precision,
                "macro_recall": result.macro_recall,
                "macro_f1": result.macro_f1,
                "train_time_s": result.train_time_s,
                "mean_inference_ms": result.mean_inference_ms,
                "artifact_size_mb": result.artifact_size_mb,
            }
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
                writer.writeheader()
                writer.writerow(flat)
            logger.info("Metrics CSV saved → %s", csv_path)
        except Exception as exc:
            logger.error("Failed to write CSV metrics for '%s': %s", result.pipeline, exc)
