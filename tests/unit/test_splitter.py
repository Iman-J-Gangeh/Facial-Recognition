"""
tests/unit/test_splitter.py

Unit tests for ``src.preprocessing.splitter`` — covers Splitter, SplitIndex,
and all acceptance criteria from Requirements 3.1–3.8 / 17.4.
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from src.preprocessing.splitter import Sample, SplitIndex, Splitter
from src.utils.config_loader import Config
from src.utils.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    *,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_seed: int = 42,
    split_metadata_path: str = "",   # filled in per test
    force_resplit: bool = False,
) -> Config:
    """Return a minimal Config with split-related fields populated."""
    return Config(
        dataset_path="data/kaggle",
        dataset_type="kaggle",
        max_identities=None,
        max_images_per_identity=None,
        detector_backend="haar",
        no_face_fallback="skip",
        image_size=(224, 224),
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        split_ratios=split_ratios,
        random_seed=random_seed,
        split_metadata_path=split_metadata_path,
        force_resplit=force_resplit,
        vocab_size=100,
        kmeans_max_iter=100,
        knn_k=5,
        knn_metric="euclidean",
        bovw_artifact_path="outputs/artifacts/classical/kmeans_vocab.pkl",
        tfidf_artifact_path="outputs/artifacts/classical/tfidf_transformer.pkl",
        knn_artifact_path="outputs/artifacts/classical/knn_classifier.pkl",
        retrain=False,
        verification_mode=False,
        verification_threshold=0.5,
        ransac_enabled=False,
        architecture="resnet18",
        pretrained=False,
        epochs=1,
        optimizer="adam",
        learning_rate=0.001,
        batch_size=16,
        checkpoint_path="outputs/artifacts/deep/best_checkpoint.pt",
        eval_mode="classification",
        embedding_classifier="knn",
        cosine_threshold=0.6,
        roc_enabled=False,
        output_dir="outputs",
        plots_dir="outputs/plots",
        results_dir="outputs/results",
        artifacts_dir="outputs/artifacts/classical",
        arcface_enabled=False,
        arcface_margin=0.5,
        arcface_scale=64.0,
    )


def _make_samples(
    n_identities: int = 5,
    images_per_identity: int = 6,
) -> list[Sample]:
    """Build a synthetic list of Sample objects."""
    samples: list[Sample] = []
    for i in range(n_identities):
        identity = f"person_{i:02d}"
        for j in range(images_per_identity):
            samples.append(Sample(path=f"data/{identity}/img_{j:03d}.jpg", label=identity))
    return samples


# ---------------------------------------------------------------------------
# R3.1 / R3.6: Ratio validation
# ---------------------------------------------------------------------------

class TestRatioValidation:
    def test_valid_ratios_do_not_raise(self, tmp_path):
        cfg = _make_config(
            split_ratios=(0.7, 0.15, 0.15),
            split_metadata_path=str(tmp_path / "split.csv"),
        )
        splitter = Splitter(cfg)
        samples = _make_samples()
        # Should not raise
        result = splitter.split(samples)
        assert isinstance(result, SplitIndex)

    @pytest.mark.parametrize("ratios", [
        (0.0, 0.5, 0.5),    # train = 0
        (0.5, 0.0, 0.5),    # val = 0
        (0.5, 0.5, 0.0),    # test = 0
        (-0.1, 0.6, 0.5),   # train negative
        (0.5, 0.5, 0.5),    # sum > 1
        (0.2, 0.2, 0.2),    # sum < 1
    ])
    def test_invalid_ratios_raise_config_error(self, tmp_path, ratios):
        cfg = _make_config(
            split_ratios=ratios,
            split_metadata_path=str(tmp_path / "split.csv"),
        )
        splitter = Splitter(cfg)
        with pytest.raises(ConfigError):
            splitter.split(_make_samples())

    def test_error_message_contains_invalid_values(self, tmp_path):
        cfg = _make_config(
            split_ratios=(0.0, 0.5, 0.5),
            split_metadata_path=str(tmp_path / "split.csv"),
        )
        splitter = Splitter(cfg)
        with pytest.raises(ConfigError, match="train=0"):
            splitter.split(_make_samples())


# ---------------------------------------------------------------------------
# R3.1: Complete partition — no duplicates, union = full set
# ---------------------------------------------------------------------------

class TestCompletePartition:
    def test_union_equals_full_input(self, tmp_path):
        samples = _make_samples(n_identities=5, images_per_identity=6)
        cfg = _make_config(split_metadata_path=str(tmp_path / "split.csv"))
        result = Splitter(cfg).split(samples)

        all_paths = {s.path for s in samples}
        split_paths = (
            {s.path for s in result.train}
            | {s.path for s in result.val}
            | {s.path for s in result.test}
        )
        assert split_paths == all_paths

    def test_no_sample_appears_in_multiple_splits(self, tmp_path):
        samples = _make_samples(n_identities=5, images_per_identity=6)
        cfg = _make_config(split_metadata_path=str(tmp_path / "split.csv"))
        result = Splitter(cfg).split(samples)

        train_paths = {s.path for s in result.train}
        val_paths = {s.path for s in result.val}
        test_paths = {s.path for s in result.test}

        assert train_paths.isdisjoint(val_paths), "Train and val overlap!"
        assert train_paths.isdisjoint(test_paths), "Train and test overlap!"
        assert val_paths.isdisjoint(test_paths), "Val and test overlap!"

    def test_total_count_preserved(self, tmp_path):
        samples = _make_samples(n_identities=4, images_per_identity=5)
        cfg = _make_config(split_metadata_path=str(tmp_path / "split.csv"))
        result = Splitter(cfg).split(samples)

        total = len(result.train) + len(result.val) + len(result.test)
        assert total == len(samples)


# ---------------------------------------------------------------------------
# R3.3 / R3.8: Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self, tmp_path):
        samples = _make_samples(n_identities=5, images_per_identity=6)

        cfg1 = _make_config(
            split_metadata_path=str(tmp_path / "split1.csv"),
            random_seed=42,
            force_resplit=True,
        )
        cfg2 = _make_config(
            split_metadata_path=str(tmp_path / "split2.csv"),
            random_seed=42,
            force_resplit=True,
        )

        result1 = Splitter(cfg1).split(samples)
        result2 = Splitter(cfg2).split(samples)

        assert sorted(s.path for s in result1.train) == sorted(s.path for s in result2.train)
        assert sorted(s.path for s in result1.val) == sorted(s.path for s in result2.val)
        assert sorted(s.path for s in result1.test) == sorted(s.path for s in result2.test)

    def test_different_seeds_may_differ(self, tmp_path):
        samples = _make_samples(n_identities=5, images_per_identity=10)

        cfg1 = _make_config(
            split_metadata_path=str(tmp_path / "split1.csv"),
            random_seed=0,
            force_resplit=True,
        )
        cfg2 = _make_config(
            split_metadata_path=str(tmp_path / "split2.csv"),
            random_seed=99999,
            force_resplit=True,
        )

        result1 = Splitter(cfg1).split(samples)
        result2 = Splitter(cfg2).split(samples)

        # With different seeds the assignments *may* differ (not guaranteed but
        # extremely likely for non-trivial datasets)
        train_paths1 = sorted(s.path for s in result1.train)
        train_paths2 = sorted(s.path for s in result2.train)
        # We cannot assert they always differ, but we assert both are valid.
        assert len(result1.train) > 0
        assert len(result2.train) > 0


# ---------------------------------------------------------------------------
# R3.4 / R3.5: Save and load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_csv_with_correct_columns(self, tmp_path):
        samples = _make_samples()
        csv_path = tmp_path / "split.csv"
        cfg = _make_config(split_metadata_path=str(csv_path))
        result = Splitter(cfg).split(samples)

        assert csv_path.exists()
        df = __import__("pandas").read_csv(csv_path)
        assert set(df.columns) == {"path", "label", "split"}

    def test_load_recovers_identical_split(self, tmp_path):
        samples = _make_samples()
        csv_path = tmp_path / "split.csv"
        cfg = _make_config(split_metadata_path=str(csv_path), force_resplit=True)
        splitter = Splitter(cfg)
        original = splitter.split(samples)
        loaded = splitter.load()

        assert sorted(s.path for s in original.train) == sorted(s.path for s in loaded.train)
        assert sorted(s.path for s in original.val) == sorted(s.path for s in loaded.val)
        assert sorted(s.path for s in original.test) == sorted(s.path for s in loaded.test)

    def test_second_split_call_loads_from_disk(self, tmp_path):
        """Without force_resplit the second call should load, not recompute."""
        samples = _make_samples()
        csv_path = tmp_path / "split.csv"
        cfg = _make_config(split_metadata_path=str(csv_path), force_resplit=False)
        splitter = Splitter(cfg)

        first = splitter.split(samples)
        # Metadata now exists → second call should load
        second = splitter.split(samples)

        assert sorted(s.path for s in first.train) == sorted(s.path for s in second.train)

    def test_force_resplit_ignores_existing_metadata(self, tmp_path):
        """With force_resplit=True a new split is computed even if CSV exists."""
        samples = _make_samples(n_identities=5, images_per_identity=6)
        csv_path = tmp_path / "split.csv"

        # First run to create CSV
        cfg_first = _make_config(
            split_metadata_path=str(csv_path),
            force_resplit=False,
            random_seed=0,
        )
        Splitter(cfg_first).split(samples)

        # Second run with different seed + force_resplit=True
        cfg_second = _make_config(
            split_metadata_path=str(csv_path),
            force_resplit=True,
            random_seed=12345,
        )
        second = Splitter(cfg_second).split(samples)

        # Result is valid
        total = len(second.train) + len(second.val) + len(second.test)
        assert total == len(samples)

    def test_csv_split_column_values(self, tmp_path):
        samples = _make_samples()
        csv_path = tmp_path / "split.csv"
        cfg = _make_config(split_metadata_path=str(csv_path))
        Splitter(cfg).split(samples)

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert set(df["split"].unique()).issubset({"train", "validation", "test"})


# ---------------------------------------------------------------------------
# R3.7 / 17.4: Identities with < 3 images → all to train + WARNING
# ---------------------------------------------------------------------------

class TestSmallIdentities:
    def test_small_identity_all_in_train(self, tmp_path, caplog):
        # Identity "small" has only 2 images (< 3)
        samples = (
            _make_samples(n_identities=4, images_per_identity=6)  # normal identities
            + [
                Sample(path="data/small/img_000.jpg", label="small"),
                Sample(path="data/small/img_001.jpg", label="small"),
            ]
        )
        csv_path = tmp_path / "split.csv"
        cfg = _make_config(split_metadata_path=str(csv_path))

        import logging
        with caplog.at_level(logging.WARNING, logger="facial_recognition"):
            result = Splitter(cfg).split(samples)

        # All "small" images must be in train
        small_in_train = [s for s in result.train if s.label == "small"]
        small_in_val = [s for s in result.val if s.label == "small"]
        small_in_test = [s for s in result.test if s.label == "small"]

        assert len(small_in_train) == 2
        assert len(small_in_val) == 0
        assert len(small_in_test) == 0

    def test_small_identity_warning_logged(self, tmp_path, caplog):
        samples = (
            _make_samples(n_identities=4, images_per_identity=6)
            + [Sample(path="data/tiny/img_0.jpg", label="tiny")]
        )
        csv_path = tmp_path / "split.csv"
        cfg = _make_config(split_metadata_path=str(csv_path))

        import logging
        with caplog.at_level(logging.WARNING, logger="facial_recognition"):
            Splitter(cfg).split(samples)

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("tiny" in msg for msg in warning_msgs), (
            "Expected a WARNING mentioning the small identity name 'tiny'"
        )

    def test_single_image_identity_goes_to_train(self, tmp_path):
        samples = (
            _make_samples(n_identities=3, images_per_identity=5)
            + [Sample(path="data/solo/img_0.jpg", label="solo")]
        )
        csv_path = tmp_path / "split.csv"
        cfg = _make_config(split_metadata_path=str(csv_path))
        result = Splitter(cfg).split(samples)

        assert any(s.label == "solo" for s in result.train)
        assert not any(s.label == "solo" for s in result.val)
        assert not any(s.label == "solo" for s in result.test)


# ---------------------------------------------------------------------------
# Size sanity checks
# ---------------------------------------------------------------------------

class TestSizeSanity:
    def test_splits_are_non_empty(self, tmp_path):
        samples = _make_samples(n_identities=5, images_per_identity=6)
        cfg = _make_config(split_metadata_path=str(tmp_path / "split.csv"))
        result = Splitter(cfg).split(samples)

        assert len(result.train) > 0
        assert len(result.val) > 0
        assert len(result.test) > 0

    def test_train_is_largest_split(self, tmp_path):
        samples = _make_samples(n_identities=5, images_per_identity=10)
        cfg = _make_config(
            split_ratios=(0.7, 0.15, 0.15),
            split_metadata_path=str(tmp_path / "split.csv"),
        )
        result = Splitter(cfg).split(samples)

        assert len(result.train) > len(result.val)
        assert len(result.train) > len(result.test)

    def test_labels_preserved_in_all_splits(self, tmp_path):
        samples = _make_samples(n_identities=4, images_per_identity=6)
        cfg = _make_config(split_metadata_path=str(tmp_path / "split.csv"))
        result = Splitter(cfg).split(samples)

        original_labels = {s.label for s in samples}
        # Every identity that appeared in the original set should appear in
        # at least one split
        split_labels = (
            {s.label for s in result.train}
            | {s.label for s in result.val}
            | {s.label for s in result.test}
        )
        assert original_labels == split_labels
