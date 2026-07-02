"""
tests/unit/test_dataset_loader.py

Unit tests for src/preprocessing/dataset_loader.py — DatasetLoader.

Covers requirements R1.1–R1.8.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.preprocessing.dataset_loader import DatasetLoader, Sample, _SUPPORTED_DATASET_TYPES
from src.utils.config_loader import Config
from src.utils.exceptions import DatasetError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    dataset_path: str,
    dataset_type: str = "kaggle",
    max_identities: int | None = None,
    max_images_per_identity: int | None = None,
) -> Config:
    """Return a minimal Config with only dataset-loader relevant fields set."""
    return Config(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        max_identities=max_identities,
        max_images_per_identity=max_images_per_identity,
        # ── preprocessing defaults ────────────────────────────────────
        detector_backend="haar",
        no_face_fallback="skip",
        image_size=(224, 224),
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        # ── splitting ─────────────────────────────────────────────────
        split_ratios=(0.7, 0.15, 0.15),
        random_seed=42,
        split_metadata_path="outputs/splits/split_index.csv",
        force_resplit=False,
        # ── classical ─────────────────────────────────────────────────
        vocab_size=100,
        kmeans_max_iter=100,
        knn_k=5,
        knn_metric="euclidean",
        bovw_artifact_path="outputs/artifacts/classical/kmeans_vocab.pkl",
        tfidf_artifact_path="outputs/artifacts/classical/tfidf_transformer.pkl",
        knn_artifact_path="outputs/artifacts/classical/knn_classifier.pkl",
        retrain=False,
        # ── verification ──────────────────────────────────────────────
        verification_mode=False,
        verification_threshold=0.5,
        ransac_enabled=False,
        # ── deep learning ─────────────────────────────────────────────
        architecture="resnet18",
        pretrained=False,
        epochs=1,
        optimizer="adam",
        learning_rate=0.001,
        batch_size=8,
        checkpoint_path="outputs/artifacts/deep/best_checkpoint.pt",
        eval_mode="classification",
        embedding_classifier="knn",
        cosine_threshold=0.6,
        arcface_enabled=False,
        arcface_margin=0.5,
        arcface_scale=64.0,
        # ── evaluation ────────────────────────────────────────────────
        roc_enabled=False,
        output_dir="outputs",
        plots_dir="outputs/plots",
        results_dir="outputs/results",
        artifacts_dir="outputs/artifacts/classical",
    )


def _create_fake_image(path: Path) -> None:
    """Write a minimal valid JPEG header so PIL can open the file."""
    # Minimal 1×1 white JPEG bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        img = Image.new("RGB", (4, 4), color=(255, 255, 255))
        img.save(str(path))
    except ImportError:
        # PIL not installed — write a dummy non-empty file
        path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 100)


def _create_dataset(tmp_path: Path, structure: dict[str, int]) -> None:
    """Create a synthetic dataset under *tmp_path*.

    *structure* maps identity name → number of JPEG images to create.
    """
    for identity, n_images in structure.items():
        identity_dir = tmp_path / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_images):
            _create_fake_image(identity_dir / f"img_{i:03d}.jpg")


# ---------------------------------------------------------------------------
# R1.2: DatasetError when path does not exist
# ---------------------------------------------------------------------------

class TestDatasetPathNotFound:
    def test_raises_dataset_error_for_missing_path(self, tmp_path):
        missing = str(tmp_path / "nonexistent")
        cfg = _make_config(dataset_path=missing)
        loader = DatasetLoader(cfg)

        with pytest.raises(DatasetError) as exc_info:
            loader.load()

        msg = str(exc_info.value)
        assert "nonexistent" in msg or "does not exist" in msg

    def test_error_message_contains_dataset_type(self, tmp_path):
        missing = str(tmp_path / "nonexistent")
        cfg = _make_config(dataset_path=missing, dataset_type="vggface2")
        loader = DatasetLoader(cfg)

        with pytest.raises(DatasetError) as exc_info:
            loader.load()

        assert "vggface2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# R1.7: DatasetError for unrecognised dataset_type
# ---------------------------------------------------------------------------

class TestUnsupportedDatasetType:
    def test_raises_for_unknown_type(self, tmp_path):
        cfg = _make_config(dataset_path=str(tmp_path), dataset_type="imagenet")
        loader = DatasetLoader(cfg)

        with pytest.raises(DatasetError) as exc_info:
            loader.load()

        msg = str(exc_info.value)
        assert "imagenet" in msg
        # Must list accepted values
        for accepted in _SUPPORTED_DATASET_TYPES:
            assert accepted in msg

    def test_raises_before_path_check(self, tmp_path):
        """Type validation should happen before path check."""
        missing_path = str(tmp_path / "no_dir")
        cfg = _make_config(dataset_path=missing_path, dataset_type="bad_type")
        loader = DatasetLoader(cfg)

        # Should raise DatasetError mentioning the bad type (not the missing path)
        with pytest.raises(DatasetError) as exc_info:
            loader.load()

        assert "bad_type" in str(exc_info.value)


# ---------------------------------------------------------------------------
# R1.1: Basic loading returns correct Sample objects
# ---------------------------------------------------------------------------

class TestBasicLoading:
    def test_returns_samples_with_correct_labels(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 3, "bob": 2})
        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)

        samples = loader.load()

        labels = {s.label for s in samples}
        assert labels == {"alice", "bob"}

    def test_returns_correct_sample_count(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 3, "bob": 2})
        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)

        samples = loader.load()
        assert len(samples) == 5

    def test_sample_paths_exist_on_disk(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 2})
        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)

        samples = loader.load()
        for s in samples:
            assert Path(s.path).exists()

    def test_returns_list_of_sample_instances(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 1})
        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)

        samples = loader.load()
        assert all(isinstance(s, Sample) for s in samples)

    def test_supports_kaggle_type(self, tmp_path):
        _create_dataset(tmp_path, {"person1": 2})
        cfg = _make_config(dataset_path=str(tmp_path), dataset_type="kaggle")
        loader = DatasetLoader(cfg)
        samples = loader.load()
        assert len(samples) == 2

    def test_supports_vggface2_type(self, tmp_path):
        _create_dataset(tmp_path, {"n000001": 2})
        cfg = _make_config(dataset_path=str(tmp_path), dataset_type="vggface2")
        loader = DatasetLoader(cfg)
        samples = loader.load()
        assert len(samples) == 2

    def test_empty_dataset_dir_returns_empty_list(self, tmp_path):
        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)
        samples = loader.load()
        assert samples == []


# ---------------------------------------------------------------------------
# R1.4: max_identities cap
# ---------------------------------------------------------------------------

class TestMaxIdentities:
    def test_caps_number_of_identities(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 2, "bob": 2, "carol": 2})
        cfg = _make_config(dataset_path=str(tmp_path), max_identities=2)
        loader = DatasetLoader(cfg)

        samples = loader.load()
        unique_labels = {s.label for s in samples}
        assert len(unique_labels) <= 2

    def test_max_identities_one(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 3, "bob": 3})
        cfg = _make_config(dataset_path=str(tmp_path), max_identities=1)
        loader = DatasetLoader(cfg)

        samples = loader.load()
        unique_labels = {s.label for s in samples}
        assert len(unique_labels) == 1

    def test_max_identities_none_loads_all(self, tmp_path):
        _create_dataset(tmp_path, {"a": 2, "b": 2, "c": 2})
        cfg = _make_config(dataset_path=str(tmp_path), max_identities=None)
        loader = DatasetLoader(cfg)

        samples = loader.load()
        assert {s.label for s in samples} == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# R1.5: max_images_per_identity cap
# ---------------------------------------------------------------------------

class TestMaxImagesPerIdentity:
    def test_caps_images_per_identity(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 5})
        cfg = _make_config(dataset_path=str(tmp_path), max_images_per_identity=3)
        loader = DatasetLoader(cfg)

        samples = loader.load()
        alice_count = sum(1 for s in samples if s.label == "alice")
        assert alice_count <= 3

    def test_cap_applied_per_identity_independently(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 5, "bob": 5})
        cfg = _make_config(dataset_path=str(tmp_path), max_images_per_identity=2)
        loader = DatasetLoader(cfg)

        samples = loader.load()
        for label in ("alice", "bob"):
            count = sum(1 for s in samples if s.label == label)
            assert count <= 2

    def test_max_images_none_loads_all(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 4})
        cfg = _make_config(dataset_path=str(tmp_path), max_images_per_identity=None)
        loader = DatasetLoader(cfg)

        samples = loader.load()
        assert len(samples) == 4

    def test_cap_larger_than_available_keeps_all(self, tmp_path):
        _create_dataset(tmp_path, {"alice": 2})
        cfg = _make_config(dataset_path=str(tmp_path), max_images_per_identity=100)
        loader = DatasetLoader(cfg)

        samples = loader.load()
        assert len(samples) == 2


# ---------------------------------------------------------------------------
# R1.3: Skip corrupt/unreadable images with WARNING
# ---------------------------------------------------------------------------

class TestCorruptImages:
    def test_skips_empty_file_when_pil_unavailable(self, tmp_path, caplog):
        """An empty file should be skipped and a WARNING logged."""
        identity_dir = tmp_path / "alice"
        identity_dir.mkdir()
        empty_file = identity_dir / "empty.jpg"
        empty_file.write_bytes(b"")

        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)

        # Patch PIL to be unavailable so fallback path is exercised
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            import importlib
            import sys
            # Remove PIL from sys.modules to trigger ImportError
            pil_modules = [k for k in sys.modules if k.startswith("PIL")]
            saved = {k: sys.modules.pop(k) for k in pil_modules}
            try:
                with caplog.at_level("WARNING"):
                    samples = loader.load()
            finally:
                sys.modules.update(saved)

        # Empty file should be excluded
        assert not any(s.path == str(empty_file) for s in samples)

    def test_warns_and_excludes_corrupt_image_with_pil(self, tmp_path, caplog):
        """Corrupt image bytes should be skipped with a WARNING."""
        identity_dir = tmp_path / "alice"
        identity_dir.mkdir()
        # Valid image
        _create_fake_image(identity_dir / "good.jpg")
        # Corrupt image (not a valid image)
        bad_file = identity_dir / "corrupt.jpg"
        bad_file.write_bytes(b"\xff\xd8\xff" + b"\xde\xad\xbe\xef" * 20)

        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)

        try:
            from PIL import Image  # noqa: F401 – verify PIL is available
            pil_available = True
        except ImportError:
            pil_available = False

        with caplog.at_level("WARNING"):
            samples = loader.load()

        if pil_available:
            # The corrupt file must not appear in samples
            assert not any(s.path == str(bad_file) for s in samples)
            # The good image must be present
            assert any("good.jpg" in s.path for s in samples)


# ---------------------------------------------------------------------------
# R1.8: Identity with zero loadable images is excluded with WARNING
# ---------------------------------------------------------------------------

class TestEmptyIdentityExclusion:
    def test_excludes_identity_with_no_images(self, tmp_path, caplog):
        """An identity subdirectory with no image files should be excluded."""
        # bob has no images (just a text file)
        bob_dir = tmp_path / "bob"
        bob_dir.mkdir()
        (bob_dir / "readme.txt").write_text("no images here")

        # alice has valid images
        _create_dataset(tmp_path, {"alice": 2})

        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)

        with caplog.at_level("WARNING"):
            samples = loader.load()

        labels = {s.label for s in samples}
        assert "alice" in labels
        assert "bob" not in labels
        # Warning should mention 'bob'
        assert any("bob" in record.message for record in caplog.records)

    def test_non_image_files_are_ignored(self, tmp_path):
        """Non-image files in an identity dir should be ignored."""
        identity_dir = tmp_path / "alice"
        identity_dir.mkdir()
        _create_fake_image(identity_dir / "img.jpg")
        (identity_dir / "notes.txt").write_text("text")
        (identity_dir / "data.csv").write_text("a,b,c")

        cfg = _make_config(dataset_path=str(tmp_path))
        loader = DatasetLoader(cfg)
        samples = loader.load()

        # Only 1 image, text files must not appear in paths
        assert len(samples) == 1
        assert samples[0].path.endswith("img.jpg")
