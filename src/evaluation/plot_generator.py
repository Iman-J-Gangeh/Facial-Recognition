"""
src/evaluation/plot_generator.py

Generates comparison plots for both pipelines.

Requirements: 13.1–13.6
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.metrics_computer import MetricsResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PlotGenerator:
    """Generates and saves PNG comparison plots to outputs/plots/."""

    def __init__(self, plots_dir: str = "outputs/plots") -> None:
        self._plots_dir = Path(plots_dir)

    def generate_comparison_plots(
        self,
        classical: MetricsResult | None,
        deep: MetricsResult | None,
    ) -> None:
        """Generate all six comparison plots.

        If metric data is missing for a plot, logs WARNING and skips
        that plot without raising (R13.6).

        Args:
            classical: MetricsResult for the classical pipeline (or None).
            deep: MetricsResult for the deep learning pipeline (or None).
        """
        self._plots_dir.mkdir(parents=True, exist_ok=True)

        self._bar_chart(
            metric_name="Top-1 Accuracy",
            values={"classical": getattr(classical, "top1", None),
                    "deep": getattr(deep, "top1", None)},
            filename="accuracy_comparison.png",
            ylabel="Top-1 Accuracy",
        )

        self._bar_chart(
            metric_name="Macro F1-Score",
            values={"classical": getattr(classical, "macro_f1", None),
                    "deep": getattr(deep, "macro_f1", None)},
            filename="f1_comparison.png",
            ylabel="Macro F1-Score",
        )

        self._bar_chart(
            metric_name="Training Time (seconds)",
            values={"classical": getattr(classical, "train_time_s", None),
                    "deep": getattr(deep, "train_time_s", None)},
            filename="training_time_comparison.png",
            ylabel="Time (s)",
        )

        self._bar_chart(
            metric_name="Mean Inference Time (ms/image)",
            values={"classical": getattr(classical, "mean_inference_ms", None),
                    "deep": getattr(deep, "mean_inference_ms", None)},
            filename="inference_time_comparison.png",
            ylabel="Time (ms)",
        )

        if classical is not None and classical.confusion_matrix is not None:
            self._confusion_matrix_plot(classical.confusion_matrix, "classical")
        else:
            logger.warning("Missing confusion matrix data for 'classical' — skipping plot.")

        if deep is not None and deep.confusion_matrix is not None:
            self._confusion_matrix_plot(deep.confusion_matrix, "deep")
        else:
            logger.warning("Missing confusion matrix data for 'deep' — skipping plot.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bar_chart(
        self,
        metric_name: str,
        values: dict[str, float | None],
        filename: str,
        ylabel: str,
    ) -> None:
        """Generate a side-by-side bar chart and save as PNG."""
        available = {k: v for k, v in values.items() if v is not None}
        if not available:
            logger.warning(
                "Missing metric data for plot '%s' — skipping.", filename
            )
            return

        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt

            pipeline_names = list(available.keys())
            metric_values = list(available.values())
            colors = ["#4C72B0", "#DD8452"][:len(pipeline_names)]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(pipeline_names, metric_values, color=colors, width=0.4, edgecolor="black")

            for bar, val in zip(bars, metric_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max(metric_values) * 0.01,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9,
                )

            ax.set_title(metric_name, fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(metric_values) * 1.15 + 1e-9)
            plt.tight_layout()

            out_path = self._plots_dir / filename
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            logger.info("Plot saved → %s", out_path)
        except Exception as exc:
            logger.warning("Failed to generate plot '%s': %s", filename, exc)

    def _confusion_matrix_plot(self, cm: np.ndarray, pipeline_name: str) -> None:
        """Generate a confusion matrix heatmap and save as PNG."""
        filename = f"{pipeline_name}_confusion_matrix.png"
        if cm is None or cm.size == 0:
            logger.warning("Missing confusion matrix data for '%s' — skipping.", pipeline_name)
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(6, cm.shape[0] * 0.6), max(5, cm.shape[0] * 0.5)))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar(im, ax=ax)

            # Annotate each cell with raw count
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8,
                    )

            ax.set_title(pipeline_name, fontsize=12, fontweight="bold")
            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")
            plt.tight_layout()

            out_path = self._plots_dir / filename
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            logger.info("Confusion matrix plot saved → %s", out_path)
        except Exception as exc:
            logger.warning("Failed to generate confusion matrix plot for '%s': %s", pipeline_name, exc)
