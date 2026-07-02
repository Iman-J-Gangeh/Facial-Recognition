"""
src/preprocessing/splitter.py

Stratified train/validation/test splitter.

Requirements: 3.1–3.8, 17.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import Config
from src.utils.exceptions import ConfigError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Minimum images required to appear in all three splits
# ---------------------------------------------------------------------------
_MIN_IMAGES_FOR_STRATIFY = 3


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """A single labeled image record.

    Defined here as well as (potentially) in dataset_loader.py so that
    splitter.py can be used independently.  Both definitions are identical
    and interchangeable at runtime.
    """
    path: str
    label: str


@dataclass
class SplitIndex:
    """The three-way partition of a dataset."""
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


# ---------------------------------------------------------------------------
# CSV column names
# ---------------------------------------------------------------------------
_COL_PATH = "path"
_COL_LABEL = "label"
_COL_SPLIT = "split"

_SPLIT_TRAIN = "train"
_SPLIT_VAL = "validation"
_SPLIT_TEST = "test"


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class Splitter:
    """Stratified train / validation / test splitter.

    Usage::

        splitter = Splitter(config)
        split_index = splitter.split(samples)
        splitter.save(split_index)

        # Later run:
        split_index = splitter.split(samples)   # loads from disk automatically
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, samples: list[Sample]) -> SplitIndex:
        """Return a :class:`SplitIndex` for *samples*.

        If ``config.split_metadata_path`` already exists on disk **and**
        ``config.force_resplit`` is *False*, the saved split is loaded and
        returned without recomputing.

        Otherwise the split is computed via stratified sampling and saved.

        Args:
            samples: Full list of labeled samples to partition.

        Returns:
            A :class:`SplitIndex` with ``train``, ``val``, and ``test``
            sample lists.

        Raises:
            ConfigError: If the configured split ratios are invalid (any
                         ratio ≤ 0.0 or the three ratios do not sum to 1.0).
        """
        self._validate_ratios(self._config.split_ratios)

        metadata_path = Path(self._config.split_metadata_path)

        if metadata_path.exists() and not self._config.force_resplit:
            logger.info(
                "Split metadata found at '%s' — loading saved split.",
                metadata_path,
            )
            return self.load()

        logger.info("Computing stratified split…")
        split_index = self._compute_split(samples)
        self.save(split_index)
        return split_index

    def save(self, split_index: SplitIndex) -> None:
        """Persist *split_index* to CSV at ``config.split_metadata_path``.

        Creates parent directories as needed.

        Args:
            split_index: The split to persist.
        """
        metadata_path = Path(self._config.split_metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        for sample in split_index.train:
            rows.append({_COL_PATH: sample.path, _COL_LABEL: sample.label, _COL_SPLIT: _SPLIT_TRAIN})
        for sample in split_index.val:
            rows.append({_COL_PATH: sample.path, _COL_LABEL: sample.label, _COL_SPLIT: _SPLIT_VAL})
        for sample in split_index.test:
            rows.append({_COL_PATH: sample.path, _COL_LABEL: sample.label, _COL_SPLIT: _SPLIT_TEST})

        df = pd.DataFrame(rows, columns=[_COL_PATH, _COL_LABEL, _COL_SPLIT])
        df.to_csv(metadata_path, index=False)
        logger.info("Split metadata saved to '%s'.", metadata_path)

    def load(self) -> SplitIndex:
        """Load a previously saved :class:`SplitIndex` from CSV.

        Returns:
            The reconstructed :class:`SplitIndex`.

        Raises:
            FileNotFoundError: If the metadata CSV does not exist.
        """
        metadata_path = Path(self._config.split_metadata_path)
        df = pd.read_csv(metadata_path)

        train: list[Sample] = []
        val: list[Sample] = []
        test: list[Sample] = []

        for _, row in df.iterrows():
            sample = Sample(path=str(row[_COL_PATH]), label=str(row[_COL_LABEL]))
            split_label = str(row[_COL_SPLIT])
            if split_label == _SPLIT_TRAIN:
                train.append(sample)
            elif split_label == _SPLIT_VAL:
                val.append(sample)
            elif split_label == _SPLIT_TEST:
                test.append(sample)
            else:
                logger.warning(
                    "Unknown split label '%s' in CSV row — skipping.", split_label
                )

        logger.info(
            "Loaded split from '%s': %d train, %d val, %d test.",
            metadata_path,
            len(train),
            len(val),
            len(test),
        )
        return SplitIndex(train=train, val=val, test=test)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ratios(ratios: tuple[float, float, float]) -> None:
        """Raise :class:`ConfigError` if *ratios* are invalid.

        Conditions that trigger an error:
        - Any ratio ≤ 0.0
        - The three ratios do not sum to 1.0 (within a small tolerance)

        Args:
            ratios: The (train, val, test) ratio tuple to validate.

        Raises:
            ConfigError: Describing the invalid values.
        """
        train_r, val_r, test_r = ratios
        invalid: list[str] = []

        if train_r <= 0.0:
            invalid.append(f"train={train_r}")
        if val_r <= 0.0:
            invalid.append(f"val={val_r}")
        if test_r <= 0.0:
            invalid.append(f"test={test_r}")

        total = train_r + val_r + test_r
        if abs(total - 1.0) > 1e-6:
            invalid.append(f"sum={total:.6f} (must be 1.0)")

        if invalid:
            raise ConfigError(
                f"Invalid split ratios — {', '.join(invalid)}. "
                f"Each ratio must be > 0.0 and all three must sum to 1.0."
            )

    def _compute_split(self, samples: list[Sample]) -> SplitIndex:
        """Perform the stratified train/val/test split.

        Identities with fewer than :data:`_MIN_IMAGES_FOR_STRATIFY` images
        cannot be stratified across three splits; they are logged at WARNING
        level and all assigned to train.

        Args:
            samples: Full dataset to split.

        Returns:
            The resulting :class:`SplitIndex`.
        """
        train_r, val_r, test_r = self._config.split_ratios
        seed = self._config.random_seed

        # ------------------------------------------------------------------
        # Separate samples by identity cardinality
        # ------------------------------------------------------------------
        from collections import defaultdict
        identity_to_samples: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            identity_to_samples[sample.label].append(sample)

        normal_samples: list[Sample] = []   # identities with >= 3 images
        small_train: list[Sample] = []       # identities with < 3 images → forced to train

        for identity, id_samples in identity_to_samples.items():
            if len(id_samples) < _MIN_IMAGES_FOR_STRATIFY:
                logger.warning(
                    "Identity '%s' has %d image(s), which is fewer than the "
                    "minimum %d required to appear in all three splits — "
                    "assigning all images to train.",
                    identity,
                    len(id_samples),
                    _MIN_IMAGES_FOR_STRATIFY,
                )
                small_train.extend(id_samples)
            else:
                normal_samples.extend(id_samples)

        train_samples: list[Sample] = list(small_train)
        val_samples: list[Sample] = []
        test_samples: list[Sample] = []

        if normal_samples:
            # ------------------------------------------------------------------
            # Step 1: split off (val + test) from train
            # ------------------------------------------------------------------
            val_test_ratio = val_r + test_r   # fraction to hold out from normal samples
            paths = [s.path for s in normal_samples]
            labels = [s.label for s in normal_samples]

            # train_test_split with stratify separates train from (val+test)
            (
                train_paths,
                val_test_paths,
                train_labels,
                val_test_labels,
            ) = train_test_split(
                paths,
                labels,
                test_size=val_test_ratio,
                random_state=seed,
                stratify=labels,
            )

            train_samples.extend(
                Sample(path=p, label=l) for p, l in zip(train_paths, train_labels)
            )

            # ------------------------------------------------------------------
            # Step 2: split (val + test) into val and test
            # ------------------------------------------------------------------
            # val fraction within the (val+test) portion
            val_within_val_test = val_r / val_test_ratio

            (
                val_paths,
                test_paths,
                val_labels,
                test_labels,
            ) = train_test_split(
                val_test_paths,
                val_test_labels,
                test_size=1.0 - val_within_val_test,
                random_state=seed,
                stratify=val_test_labels,
            )

            val_samples = [Sample(path=p, label=l) for p, l in zip(val_paths, val_labels)]
            test_samples = [Sample(path=p, label=l) for p, l in zip(test_paths, test_labels)]

        logger.info(
            "Split complete: %d train, %d val, %d test (total %d).",
            len(train_samples),
            len(val_samples),
            len(test_samples),
            len(samples),
        )

        return SplitIndex(train=train_samples, val=val_samples, test=test_samples)
