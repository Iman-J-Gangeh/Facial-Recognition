"""
train.py

Train either the classical or deep learning face recognition pipeline.
"""

import argparse
import sys
import time

from src.classical.bovw_encoder import BoVWEncoder
from src.classical.knn_classifier import KNNClassifier
from src.classical.sift_extractor import SIFTExtractor
from src.classical.tfidf_transformer import TFIDFTransformer
from src.classical.vocabulary_builder import VocabularyBuilder
from src.deep.dataset import make_dataloaders
from src.deep.model_builder import ModelBuilder
from src.deep.trainer import Trainer
from src.preprocessing.face_detector import make_detector
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.splitter import Splitter
from src.utils.config_loader import ConfigLoader
from src.utils.exceptions import (
    ArtifactError,
    ConfigError,
    DatasetError,
    DetectorInitError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_classical_pipeline(config, split_index):
    logger.info("Training classical pipeline")

    start_time = time.time()

    detector = make_detector(config)
    preprocessor = Preprocessor(config, detector)

    train_samples = []
    for sample in split_index.train:
        processed = preprocessor.process_for_classical(sample)
        if processed is not None:
            train_samples.append(processed)

    if not train_samples:
        raise DatasetError("No usable training samples after preprocessing")

    logger.info("Training samples: %d", len(train_samples))

    sift = SIFTExtractor()
    descriptors = sift.extract_all(train_samples)
    logger.info("SIFT descriptors: %d", descriptors.shape[0])

    vocab_builder = VocabularyBuilder(config)
    vocabulary = vocab_builder.load_or_build(descriptors)

    bovw = BoVWEncoder(vocabulary)
    x_bovw = bovw.encode_batch(train_samples, sift)

    tfidf = TFIDFTransformer(config)
    x_train = tfidf.load_or_fit(x_bovw)

    y_train = [sample.label for sample in train_samples]

    knn = KNNClassifier(config)
    knn.load_or_train(x_train, y_train)

    total_time = time.time() - start_time

    logger.info(
        "Classical training finished in %.2f seconds",
        total_time,
    )

    metadata = {
        "pipeline": "classical",
        "training_time_s": total_time,
        "vocab_size": vocabulary.vocab_size,
        "num_training_samples": len(train_samples),
        "num_classes": len(set(y_train)),
    }

    return total_time, metadata


def train_deep_pipeline(config, split_index):
    logger.info("Training deep pipeline")

    start_time = time.time()

    train_loader, val_loader, _, idx_to_label = make_dataloaders(
        split_index,
        config,
    )

    num_classes = len(idx_to_label)
    logger.info("Classes: %d", num_classes)

    model = ModelBuilder().build(config, num_classes)

    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader)

    total_time = time.time() - start_time
    best_val_accuracy = (
        max(history.epoch_val_accuracies)
        if history.epoch_val_accuracies
        else 0.0
    )

    logger.info(
        "Deep training finished in %.2f seconds",
        total_time,
    )
    logger.info(
        "Best validation accuracy: %.4f at epoch %d",
        best_val_accuracy,
        history.best_epoch,
    )

    metadata = {
        "pipeline": "deep",
        "training_time_s": total_time,
        "architecture": config.architecture,
        "num_training_samples": len(train_loader.dataset),
        "num_classes": num_classes,
        "best_epoch": history.best_epoch,
        "best_val_accuracy": best_val_accuracy,
    }

    return total_time, metadata


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


def main():
    parser = argparse.ArgumentParser(
        description="Train a face recognition pipeline.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["classical", "deep"],
        required=True,
        help="Pipeline to train.",
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

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(split_index.train),
            len(split_index.val),
            len(split_index.test),
        )

        if args.pipeline == "classical":
            train_classical_pipeline(config, split_index)
        else:
            train_deep_pipeline(config, split_index)

        logger.info("Training complete")
        return 0

    except (ConfigError, DatasetError, DetectorInitError, ArtifactError) as exc:
        logger.error("Training failed: %s", exc)
        return 1

    except Exception as exc:
        logger.error("Unexpected error during training: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
