"""
compare.py

Run the classical and deep learning pipelines on the same test split and
write comparison plots/results.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.classical.bovw_encoder import BoVWEncoder
from src.classical.knn_classifier import KNNClassifier
from src.classical.sift_extractor import SIFTExtractor
from src.deep.dataset import get_deep_dataset
from src.deep.evaluator import ClassificationEvaluator, EmbeddingEvaluator
from src.evaluation.metrics_computer import MetricsComputer
from src.evaluation.plot_generator import PlotGenerator
from src.preprocessing.face_detector import make_detector
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.splitter import Splitter
from src.utils.artifact_manager import ArtifactManager
from src.utils.config_loader import ConfigLoader
from src.utils.exceptions import ArtifactNotFoundError, ConfigError, DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_classical(config, split_index):
    logger.info("Evaluating classical pipeline")

    artifacts = ArtifactManager()

    try:
        vocabulary = artifacts.load(config.bovw_artifact_path)
        tfidf = artifacts.load(config.tfidf_artifact_path)
        knn_payload = artifacts.load(config.knn_artifact_path)
    except ArtifactNotFoundError as exc:
        logger.error("Classical artifacts are missing. Run train.py --pipeline classical first.")
        raise exc

    knn = KNNClassifier(config)
    knn._clf = knn_payload["clf"]
    knn._classes_ = knn_payload.get("classes", [])

    detector = make_detector(config)
    preprocessor = Preprocessor(config, detector)

    test_samples = [
        processed
        for sample in split_index.test
        if (processed := preprocessor.process_for_classical(sample)) is not None
    ]

    logger.info("Classical test samples: %d", len(test_samples))

    sift = SIFTExtractor()
    bovw = BoVWEncoder(vocabulary)

    x_bovw = bovw.encode_batch(test_samples, sift)
    x_test = tfidf.transform(x_bovw)

    y_true = [sample.label for sample in test_samples]
    topk_preds = knn.predict_topk(x_test, k=1)
    y_pred = [preds[0] if preds else "UNKNOWN" for preds in topk_preds]

    labels = sorted(set(y_true + y_pred))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    scores = np.zeros((len(y_pred), len(labels)), dtype=np.float32)
    for row, pred in enumerate(y_pred):
        if pred in label_to_idx:
            scores[row, label_to_idx[pred]] = 1.0

    return MetricsComputer().compute(
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        pipeline_name="classical",
        train_time_s=0.0,
        mean_inference_ms=0.1,
    )


def evaluate_deep(config, split_index):
    logger.info("Evaluating deep pipeline")

    detector = make_detector(config)
    preprocessor = Preprocessor(config, detector)

    _, test_loader, idx_to_label = get_deep_dataset(
        config,
        split_index,
        preprocessor,
        "test",
    )

    from src.deep.model_builder import ModelBuilder

    model = ModelBuilder().build(config, len(idx_to_label))

    if config.eval_mode.lower() == "classification":
        evaluator = ClassificationEvaluator()
        output = evaluator.evaluate(model, test_loader, config, idx_to_label)
    else:
        train_loader, _, _ = get_deep_dataset(
            config,
            split_index,
            preprocessor,
            "train",
        )

        evaluator = EmbeddingEvaluator()
        output = evaluator.evaluate(
            model,
            train_loader,
            test_loader,
            config,
            idx_to_label,
        )

    return MetricsComputer().compute(
        y_true=output.y_true,
        y_pred=output.y_pred,
        scores=output.scores,
        pipeline_name="deep",
        train_time_s=0.0,
        mean_inference_ms=output.inference_time_ms,
    )


def parse_overrides(raw_args):
    overrides = {}

    for arg in raw_args:
        if not arg.startswith("--"):
            continue

        option = arg[2:]

        if "=" in option:
            key, value = option.split("=", 1)
            overrides[key] = value
        else:
            overrides[option] = "true"

    return overrides


def metrics_to_dict(metrics):
    return {
        "top1_accuracy": metrics.top1,
        "top5_accuracy": metrics.top5,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "training_time_s": metrics.train_time_s,
        "inference_time_ms": metrics.mean_inference_ms,
        "artifact_size_mb": metrics.artifact_size_mb,
    }


def write_report(config, classical_metrics, deep_metrics):
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "comparison_type": "classical_vs_deep",
        "classical": metrics_to_dict(classical_metrics),
        "deep": metrics_to_dict(deep_metrics),
    }

    output_path = results_dir / "comparison_report.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare the classical and deep facial recognition pipelines.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the config file.",
    )

    args, raw_overrides = parser.parse_known_args()
    overrides = parse_overrides(raw_overrides)

    try:
        logger.info("Loading config: %s", args.config)

        config = ConfigLoader().load(args.config, overrides)
        split_index = Splitter(config).load()

        logger.info("Running pipeline comparison")

        classical_metrics = evaluate_classical(config, split_index)
        logger.info(
            "Classical results: Top-1 %.4f, F1 %.4f",
            classical_metrics.top1,
            classical_metrics.macro_f1,
        )

        deep_metrics = evaluate_deep(config, split_index)
        logger.info(
            "Deep results: Top-1 %.4f, F1 %.4f",
            deep_metrics.top1,
            deep_metrics.macro_f1,
        )

        PlotGenerator().generate_comparison_plots(classical_metrics, deep_metrics)
        report_path = write_report(config, classical_metrics, deep_metrics)

        winner = "deep" if deep_metrics.top1 > classical_metrics.top1 else "classical"

        logger.info("Comparison complete")
        logger.info("Winner: %s", winner)
        logger.info("Plots: %s", config.plots_dir)
        logger.info("Report: %s", report_path)

        return 0

    except (ConfigError, DatasetError, ArtifactNotFoundError) as exc:
        logger.error("Comparison failed: %s", exc)
        return 1

    except Exception as exc:
        logger.error("Unexpected comparison error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())