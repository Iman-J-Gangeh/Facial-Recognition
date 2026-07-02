"""
src/deep/evaluator.py

Deep learning pipeline evaluators (classification and embedding modes).

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.exceptions import ArtifactNotFoundError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config_loader import Config

logger = get_logger(__name__)


@dataclass
class EvaluationOutput:
    """Results from a pipeline evaluation run."""
    y_true: list[str]
    y_pred: list[str]
    scores: np.ndarray        # (N, num_classes) probability/similarity matrix
    inference_time_ms: float  # mean per-image inference time in milliseconds
    idx_to_label: dict[int, str] = field(default_factory=dict)


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load the best checkpoint into the model in-place.

    Raises:
        ArtifactNotFoundError: If the checkpoint file does not exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"Checkpoint not found at '{path}'. "
            "Run 'train.py --pipeline deep' first to generate it."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_acc = checkpoint.get("val_acc", "?")
    epoch = checkpoint.get("epoch", "?")
    logger.info("Loaded checkpoint from '%s' (epoch=%s, val_acc=%s).", path, epoch, val_acc)


class ClassificationEvaluator:
    """Evaluates a model using the softmax classification head (R11.1)."""

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: "Config",
        idx_to_label: dict[int, str],
    ) -> EvaluationOutput:
        """Run classification evaluation.

        Args:
            model: The trained nn.Module.
            test_loader: DataLoader for the test split.
            config: Validated Config instance.
            idx_to_label: Mapping from class index to label string.

        Returns:
            EvaluationOutput with predictions, softmax scores, and timing.
        """
        # Load best checkpoint (R11.5)
        _load_checkpoint(model, config.checkpoint_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        y_true: list[str] = []
        y_pred: list[str] = []
        all_scores: list[np.ndarray] = []
        total_time = 0.0
        total_images = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                t0 = time.perf_counter()
                logits = model(imgs)
                elapsed = time.perf_counter() - t0
                total_time += elapsed
                total_images += imgs.size(0)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

                all_scores.append(probs)
                for true_idx, pred_idx in zip(labels.numpy(), preds):
                    y_true.append(idx_to_label.get(int(true_idx), str(true_idx)))
                    y_pred.append(idx_to_label.get(int(pred_idx), str(pred_idx)))

        mean_inference_ms = (total_time / max(total_images, 1)) * 1000.0
        scores_matrix = np.vstack(all_scores) if all_scores else np.empty((0, 0))

        return EvaluationOutput(
            y_true=y_true,
            y_pred=y_pred,
            scores=scores_matrix,
            inference_time_ms=mean_inference_ms,
            idx_to_label=idx_to_label,
        )


class EmbeddingEvaluator:
    """Evaluates a model by extracting embedding layers"""

    def evaluate(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: "Config",
        idx_to_label: dict[int, str],
    ) -> EvaluationOutput:
        """Run embedding-based evaluation.

        Args:
            model: The trained nn.Module.
            train_loader: DataLoader for training data (used to build reference embeddings).
            test_loader: DataLoader for test data.
            config: Validated Config instance.
            idx_to_label: Mapping from class index to label string.

        Returns:
            EvaluationOutput with predictions, similarity scores, and timing.
        """
        # Load best checkpoint (R11.5)
        _load_checkpoint(model, config.checkpoint_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Extract embeddings
        logger.info("Extracting training embeddings...")
        train_embeddings, train_labels = self._extract_embeddings(model, train_loader, device)

        logger.info("Extracting test embeddings...")
        t0 = time.perf_counter()
        test_embeddings, test_labels_idx = self._extract_embeddings(model, test_loader, device)
        embed_time = time.perf_counter() - t0
        mean_inference_ms = (embed_time / max(len(test_embeddings), 1)) * 1000.0

        classifier = config.embedding_classifier.lower()
        label_strings = sorted(idx_to_label.values())
        num_classes = len(label_strings)

        if classifier == "knn":
            y_pred, scores = self._knn_classify(
                train_embeddings, train_labels, test_embeddings, label_strings, idx_to_label
            )
        else:  # cosine
            y_pred, scores = self._cosine_classify(
                train_embeddings, train_labels, test_embeddings,
                label_strings, idx_to_label, config.cosine_threshold
            )

        y_true = [idx_to_label.get(int(i), str(i)) for i in test_labels_idx]

        return EvaluationOutput(
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
            inference_time_ms=mean_inference_ms,
            idx_to_label=idx_to_label,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_embeddings(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract penultimate-layer embeddings from all batches."""
        embeddings_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

        # Hook to capture penultimate layer output
        penultimate_output: list[torch.Tensor] = []

        def _hook(module, input, output):
            penultimate_output.append(output.detach().cpu())

        # Register hook on the layer before the final classifier
        hook_handle = None
        for name, module in model.named_children():
            if name in ("fc", "classifier"):
                # Get the layer before this one
                break
        # Use adaptive average pool as the penultimate layer for ResNet
        # For other architectures, try avgpool or features
        hooked_layer = None
        for attr in ("avgpool", "features", "global_average_pooling"):
            if hasattr(model, attr):
                hooked_layer = getattr(model, attr)
                break
        if hooked_layer is None:
            # Fall back: hook the final layer and use its input
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Linear):
                    hooked_layer = module
                    break

        if hooked_layer is not None:
            hook_handle = hooked_layer.register_forward_hook(_hook)

        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                penultimate_output.clear()
                _ = model(imgs)
                if penultimate_output:
                    emb = penultimate_output[0]
                    # Flatten spatial dimensions if needed
                    if emb.ndim > 2:
                        emb = emb.flatten(1)
                    embeddings_list.append(emb.numpy())
                else:
                    # No hook fired — use logits as fallback
                    pass
                labels_list.append(labels.numpy())

        if hook_handle is not None:
            hook_handle.remove()

        embeddings = np.vstack(embeddings_list) if embeddings_list else np.empty((0, 1))
        labels_arr = np.concatenate(labels_list) if labels_list else np.empty(0, dtype=int)
        return embeddings, labels_arr

    @staticmethod
    def _knn_classify(
        train_emb: np.ndarray,
        train_labels: np.ndarray,
        test_emb: np.ndarray,
        label_strings: list[str],
        idx_to_label: dict[int, str],
    ) -> tuple[list[str], np.ndarray]:
        """Classify test embeddings with KNN (k=5)."""
        from sklearn.neighbors import KNeighborsClassifier

        k = min(5, len(train_emb))
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(train_emb, train_labels)

        preds_idx = knn.predict(test_emb)
        probs = knn.predict_proba(test_emb)

        # Align proba columns to label_strings
        num_classes = len(label_strings)
        scores = np.zeros((len(test_emb), num_classes), dtype=np.float32)
        for col_idx, cls_val in enumerate(knn.classes_):
            label = idx_to_label.get(int(cls_val), str(cls_val))
            if label in label_strings:
                out_col = label_strings.index(label)
                scores[:, out_col] = probs[:, col_idx]

        y_pred = [idx_to_label.get(int(i), str(i)) for i in preds_idx]
        return y_pred, scores

    @staticmethod
    def _cosine_classify(
        train_emb: np.ndarray,
        train_labels: np.ndarray,
        test_emb: np.ndarray,
        label_strings: list[str],
        idx_to_label: dict[int, str],
        threshold: float,
    ) -> tuple[list[str], np.ndarray]:
        """Classify by finding the training embedding with the highest cosine similarity."""
        # L2-normalise
        def _norm(x):
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return x / norms

        train_n = _norm(train_emb)
        test_n = _norm(test_emb)

        sim_matrix = test_n @ train_n.T  # (N_test, N_train)
        num_classes = len(label_strings)
        scores = np.zeros((len(test_emb), num_classes), dtype=np.float32)
        y_pred: list[str] = []

        _no_match = "unknown"

        for i, row in enumerate(sim_matrix):
            best_idx = int(np.argmax(row))
            best_sim = float(row[best_idx])
            best_label_idx = int(train_labels[best_idx])
            best_label = idx_to_label.get(best_label_idx, str(best_label_idx))

            if best_sim >= threshold:
                y_pred.append(best_label)
            else:
                y_pred.append(_no_match)

            if best_label in label_strings:
                out_col = label_strings.index(best_label)
                scores[i, out_col] = best_sim

        return y_pred, scores
