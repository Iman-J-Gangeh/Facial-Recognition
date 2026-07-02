"""
preprocess.py

Prepare the dataset and save the train/validation/test split.
"""

import argparse
import sys

from src.preprocessing.dataset_loader import DatasetLoader
from src.preprocessing.face_detector import make_detector
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.splitter import Splitter
from src.utils.config_loader import ConfigLoader
from src.utils.exceptions import DatasetError, DetectorInitError, PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


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


def filter_valid_faces(samples, preprocessor):
    valid_samples = []
    skipped = 0

    progress_step = max(1, len(samples) // 10)

    for i, sample in enumerate(samples, start=1):
        processed = preprocessor.process_for_classical(sample)

        if processed is not None:
            valid_samples.append(sample)
        else:
            skipped += 1

        if i % progress_step == 0:
            logger.info("Checked %d/%d images", i, len(samples))

    return valid_samples, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the face dataset and create the data split.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--help-config",
        action="store_true",
        help="Show available config options and exit.",
    )

    args, raw_overrides = parser.parse_known_args()
    overrides = parse_overrides(raw_overrides)

    try:
        logger.info("Loading config from %s", args.config)

        config = ConfigLoader().load(args.config, overrides)

        dataset_loader = DatasetLoader(config)
        samples = dataset_loader.load()

        if not samples:
            raise DatasetError("No samples found in the dataset")

        logger.info(
            "Loaded %d samples from %d identities",
            len(samples),
            len(set(sample.label for sample in samples)),
        )

        detector = make_detector(config)
        preprocessor = Preprocessor(config, detector)

        valid_samples, skipped = filter_valid_faces(samples, preprocessor)

        logger.info(
            "Face check finished: %d usable, %d skipped",
            len(valid_samples),
            skipped,
        )

        if not valid_samples:
            raise PreprocessingError("No usable face detections found")

        splitter = Splitter(config)
        split_index = splitter.split(valid_samples)
        splitter.save(split_index)

        logger.info("Saved split metadata to %s", config.split_metadata_path)
        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(split_index.train),
            len(split_index.val),
            len(split_index.test),
        )

        logger.info("Preprocessing complete")
        return 0

    except (DatasetError, DetectorInitError, PreprocessingError) as exc:
        logger.error("Preprocessing failed: %s", exc)
        return 1

    except Exception as exc:
        logger.error("Unexpected error during preprocessing: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
