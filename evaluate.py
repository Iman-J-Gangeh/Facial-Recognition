"""
evaluate.py

Evaluate either the classical or deep learning pipeline on the test split.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.classical.bovw_encoder import BoVWEncoder
from src.classical.knn_classifier import KNNClassifier
from src.classical.sift_extractor import SIFTExtractor
from src.deep.dataset import get_deep_dataset
from src.deep.evaluator import ClassificationEvaluator, EmbeddingEvaluator
from src.evaluation.metrics_computer import MetricsComputer
from src.evaluation.report_writer import ReportWriter
from src.preprocessing.face_detector import make_detector
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.splitter import Splitter
from src.utils.artifact_manager import ArtifactManager
from src.utils.config_loader import ConfigLoader
from src.utils.exceptions import ArtifactNotFoundError, ConfigError, DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_scores(predictions, label_to_idx):
    scores = np.zeros((len(predictions), len(label_to_idx)), dtype=np.float32)

    for row, pred in enumerate(predictions):
        if pred in label_to_idx:
            scores[row, label_to_idx[pred]] = 1.0

    return scores


def evaluate_classical_pipeline(config, split_index):
    logger.info("Evaluating classical pipeline")

    start_time = time.time()
    artifacts = ArtifactManager()

    try:
        vocabulary = artifacts.load(
            config.bovw_artifact_path,
            expected_meta={"vocab_size": config.vocab_size},
        )
        tfidf_transformer = artifacts.load(config.tfidf_artifact_path)
        knn_payload = artifacts.load(config.knn_artifact_path)
    except ArtifactNotFoundError as exc:
        config_path = getattr(config, "config_path", getattr(config, "config_yaml", "config.yaml"))
        logger.error(
            "Could not find the classical pipeline artifacts. "
            "Run: python train.py --pipeline classical --config %s",
            config_path,
        )
        raise exc

    knn = KNNClassifier(config)
    knn._clf = knn_payload["clf"]
    knn._classes_ = knn_payload.get("classes", [])

    detector = make_detector(config)
    preprocessor = Preprocessor(config, detector)

    test_samples = []
    for sample in split_index.test:
        processed = preprocessor.process_for_classical(sample)
        if processed is not None:
            test_samples.append(processed)

    if not test_samples:
        raise DatasetError("No usable test samples after preprocessing")

    logger.info("Classical test samples: %d", len(test_samples))

    sift = SIFTExtractor()
    bovw = BoVWEncoder(vocabulary)

    x_bovw = bovw.encode_batch(test_samples, sift)
    x_test = tfidf_transformer.transform(x_bovw)

    y_true = [sample.label for sample in test_samples]
    topk_predictions = knn.predict_topk(x_test, k=1)
    y_pred = [preds[0] if preds else "UNKNOWN" for preds in topk_predictions]

    labels = sorted(set(y_true + y_pred))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    scores = make_scores(y_pred, label_to_idx)

    elapsed = time.time() - start_time
    mean_inference_ms = elapsed / max(len(test_samples), 1) * 1000.0

    metrics = MetricsComputer().compute(
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        pipeline_name="classical",
        train_time_s=0.0,
        mean_inference_ms=mean_inference_ms,
    )

    metrics.artifact_size_mb = artifacts.measure_size_mb(
        [
            config.bovw_artifact_path,
            config.tfidf_artifact_path,
            config.knn_artifact_path,
        ]
    )

    ReportWriter().save(metrics)

    logger.info(
        "Classical results: Top-1 %.4f, F1 %.4f, inference %.2f ms",
        metrics.top1,
        metrics.macro_f1,
        mean_inference_ms,
    )

    return metrics


def evaluate_deep_pipeline(config, split_index):
    logger.info("Evaluating deep pipeline")

    start_time = time.time()

    detector = make_detector(config)
    preprocessor = Preprocessor(config, detector)

    train_loader, test_loader, idx_to_label = get_deep_dataset(
        config,
        split_index,
        preprocessor,
        "test",
    )

    from src.deep.model_builder import ModelBuilder

    model = ModelBuilder().build(config, len(idx_to_label))
    eval_mode = config.eval_mode.lower()

    try:
        if eval_mode == "classification":
            evaluator = ClassificationEvaluator()
            output = evaluator.evaluate(model, test_loader, config, idx_to_label)
        else:
            evaluator = EmbeddingEvaluator()
            output = evaluator.evaluate(
                model,
                train_loader,
                test_loader,
                config,
                idx_to_label,
            )
    except ArtifactNotFoundError as exc:
        config_path = getattr(config, "config_path", getattr(config, "config_yaml", "config.yaml"))
        logger.error(
            "Could not find the deep model checkpoint. "
            "Run: python train.py --pipeline deep --config %s",
            config_path,
        )
        raise exc

    elapsed = time.time() - start_time
    logger.info("Deep evaluation finished in %.2f seconds", elapsed)

    metrics = MetricsComputer().compute(
        y_true=output.y_true,
        y_pred=output.y_pred,
        scores=output.scores,
        pipeline_name="deep",
        train_time_s=0.0,
        mean_inference_ms=output.inference_time_ms,
    )

    checkpoint_path = Path(config.checkpoint_path)
    metrics.artifact_size_mb = (
        checkpoint_path.stat().st_size / (1024 * 1024)
        if checkpoint_path.exists()
        else 0.0
    )

    ReportWriter().save(metrics)

    logger.info(
        "Deep results: Top-1 %.4f, F1 %.4f, inference %.2f ms",
        metrics.top1,
        metrics.macro_f1,
        output.inference_time_ms,
    )

    return metrics


def parse_overrides(raw_args):
    overrides = {}

    for arg in raw_args:
        if not arg.startswith("--"):
            continue

        arg = arg[2:]

        if "=" in arg:
            key, value = arg.split("=", 1)
            overrides[key] = value
        else:
            overrides[arg] = "true"

    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained facial recognition pipeline.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["classical", "deep"],
        required=True,
        help="Pipeline to evaluate.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the config file.",
    )

    args, raw_overrides = parser.parse_known_args()
    overrides = parse_overrides(raw_overrides)

    try:
        logger.info("Loading config from %s", args.config)

        config = ConfigLoader().load(args.config, overrides)
        split_index = Splitter(config).load()

        if args.pipeline == "classical":
            evaluate_classical_pipeline(config, split_index)
        else:
            evaluate_deep_pipeline(config, split_index)

        logger.info("Evaluation complete")
        return 0

    except (ConfigError, DatasetError, ArtifactNotFoundError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1

    except Exception as exc:
        logger.error("Unexpected error during evaluation: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
