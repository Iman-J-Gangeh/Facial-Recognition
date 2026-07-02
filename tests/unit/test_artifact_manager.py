"""
tests/unit/test_artifact_manager.py

Unit tests for ArtifactManager (task 1.4).

Covers:
- save / load round-trip for pickle artifacts
- companion manifest written alongside artifact
- ArtifactNotFoundError raised when artifact missing
- WARNING logged for each metadata field that differs from expected_meta
- measure_size_mb sums file sizes correctly
- write_run_manifest serialises the full Config
"""

from __future__ import annotations

import importlib.util as _iutil
import json
import logging
from pathlib import Path

import pytest

from src.utils.artifact_manager import ArtifactManager
from src.utils.exceptions import ArtifactError, ArtifactNotFoundError

_TORCH_AVAILABLE = _iutil.find_spec("torch") is not None

# Logger name used by ArtifactManager (derived from __name__ in that module)
_ARTM_LOGGER = "facial_recognition.src.utils.artifact_manager"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def manager() -> ArtifactManager:
    return ArtifactManager()


@pytest.fixture()
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def _enable_caplog_for_artm():
    """Context manager: temporarily enable propagation on the ArtifactManager
    logger AND its parent so pytest's caplog fixture can capture its records.

    The ``facial_recognition`` root logger has ``propagate=False`` to avoid
    double output in production.  We re-enable it in tests only.
    """
    import contextlib, logging as _logging

    @contextlib.contextmanager
    def _ctx():
        # Walk up from the specific module logger to the facial_recognition root
        loggers_to_patch = []
        name = _ARTM_LOGGER
        while name:
            lg = _logging.getLogger(name)
            loggers_to_patch.append((lg, lg.propagate))
            lg.propagate = True
            dot = name.rfind(".")
            name = name[:dot] if dot != -1 else ""
        # Also patch the top-level "facial_recognition" logger
        root_fr = _logging.getLogger("facial_recognition")
        if root_fr not in [l for l, _ in loggers_to_patch]:
            loggers_to_patch.append((root_fr, root_fr.propagate))
            root_fr.propagate = True
        try:
            yield
        finally:
            for lg, prev in loggers_to_patch:
                lg.propagate = prev

    return _ctx()


# ---------------------------------------------------------------------------
# save / load — pickle
# ---------------------------------------------------------------------------

class TestSaveLoadPickle:
    def test_round_trip_simple_object(self, manager: ArtifactManager, tmp: Path):
        obj = {"key": "value", "num": 42}
        p = tmp / "data.pkl"
        manager.save(obj, p, metadata={"pipeline": "classical"})
        result = manager.load(p)
        assert result == obj

    def test_companion_manifest_written(self, manager: ArtifactManager, tmp: Path):
        p = tmp / "data.pkl"
        meta = {"vocab_size": 1000, "architecture": "resnet18"}
        manager.save({}, p, metadata=meta)
        manifest_path = tmp / "data.pkl.manifest.json"
        assert manifest_path.exists(), "manifest file must exist next to artifact"
        with manifest_path.open() as fh:
            stored = json.load(fh)
        assert stored == meta

    def test_manifest_fields_match(self, manager: ArtifactManager, tmp: Path):
        p = tmp / "model.pkl"
        meta = {"a": 1, "b": "two"}
        manager.save({"payload": True}, p, metadata=meta)
        result = manager.load(p, expected_meta=meta)
        assert result == {"payload": True}

    def test_artifact_not_found_raises(self, manager: ArtifactManager, tmp: Path):
        with pytest.raises(ArtifactNotFoundError):
            manager.load(tmp / "nonexistent.pkl")

    def test_load_warns_on_metadata_mismatch(
        self, manager: ArtifactManager, tmp: Path, caplog
    ):
        p = tmp / "thing.pkl"
        manager.save({"x": 1}, p, metadata={"arch": "resnet18", "epochs": 10})

        with _enable_caplog_for_artm(), caplog.at_level(logging.WARNING):
            manager.load(p, expected_meta={"arch": "resnet50", "epochs": 10})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("arch" in r.message for r in warnings), (
            "Expected a WARNING mentioning the differing 'arch' field"
        )

    def test_load_warns_only_for_differing_fields(
        self, manager: ArtifactManager, tmp: Path, caplog
    ):
        p = tmp / "thing2.pkl"
        manager.save({"x": 1}, p, metadata={"a": 1, "b": 2, "c": 3})

        with _enable_caplog_for_artm(), caplog.at_level(logging.WARNING):
            manager.load(p, expected_meta={"a": 1, "b": 99, "c": 3})

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        # Only "b" differs → exactly one warning
        assert len(warnings) == 1
        assert "b" in warnings[0].message

    def test_load_no_warnings_when_meta_matches(
        self, manager: ArtifactManager, tmp: Path, caplog
    ):
        p = tmp / "exact.pkl"
        meta = {"pipeline": "classical", "vocab_size": 500}
        manager.save({"ok": True}, p, metadata=meta)

        with _enable_caplog_for_artm(), caplog.at_level(logging.WARNING):
            manager.load(p, expected_meta=meta)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0

    def test_corrupt_file_raises_artifact_error(self, manager: ArtifactManager, tmp: Path):
        p = tmp / "corrupt.pkl"
        p.write_bytes(b"this is not a pickle")
        with pytest.raises(ArtifactError):
            manager.load(p)

    def test_save_creates_parent_directories(self, manager: ArtifactManager, tmp: Path):
        p = tmp / "nested" / "deep" / "file.pkl"
        manager.save({"ok": True}, p, metadata={})
        assert p.exists()

    def test_round_trip_list_object(self, manager: ArtifactManager, tmp: Path):
        obj = list(range(1000))
        p = tmp / "list.pkl"
        manager.save(obj, p, metadata={})
        assert manager.load(p) == obj


# ---------------------------------------------------------------------------
# save / load — torch (.pt / .pth)
# Skipped when torch is not importable (BLAS / env compatibility issues).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed or incompatible")
class TestSaveLoadTorch:
    def test_torch_manifest_written_pt_extension(self, manager: ArtifactManager, tmp: Path):
        # Only test manifest creation (no actual tensor import at module level)
        import torch  # noqa: F401 — guarded by skipif
        p = tmp / "model.pt"
        t = torch.zeros(2)
        manager.save(t, p, metadata={"epoch": 5})
        assert (tmp / "model.pt.manifest.json").exists()


# ---------------------------------------------------------------------------
# measure_size_mb
# ---------------------------------------------------------------------------

class TestMeasureSizeMb:
    def test_empty_list_returns_zero(self, manager: ArtifactManager):
        assert manager.measure_size_mb([]) == pytest.approx(0.0)

    def test_single_file_one_mb(self, manager: ArtifactManager, tmp: Path):
        p = tmp / "file.bin"
        p.write_bytes(b"x" * 1024 * 1024)  # exactly 1 MiB
        result = manager.measure_size_mb([p])
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_multiple_files_summed(self, manager: ArtifactManager, tmp: Path):
        p1 = tmp / "a.bin"
        p2 = tmp / "b.bin"
        p1.write_bytes(b"a" * 512 * 1024)  # 0.5 MiB
        p2.write_bytes(b"b" * 512 * 1024)  # 0.5 MiB
        result = manager.measure_size_mb([p1, p2])
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_missing_file_skipped(self, manager: ArtifactManager, tmp: Path):
        p_exists = tmp / "real.bin"
        p_exists.write_bytes(b"y" * 1024)
        p_missing = tmp / "ghost.bin"
        result = manager.measure_size_mb([p_exists, p_missing])
        assert result == pytest.approx(1024 / (1024 * 1024), abs=1e-9)

    def test_returns_float(self, manager: ArtifactManager, tmp: Path):
        p = tmp / "f.bin"
        p.write_bytes(b"z" * 100)
        assert isinstance(manager.measure_size_mb([p]), float)

    def test_accepts_string_paths(self, manager: ArtifactManager, tmp: Path):
        p = tmp / "str_path.bin"
        p.write_bytes(b"q" * 100)
        result = manager.measure_size_mb([str(p)])
        assert result > 0.0


# ---------------------------------------------------------------------------
# write_run_manifest
# ---------------------------------------------------------------------------

_CONFIG_YAML = (
    "c:/Users/imang/OneDrive/Desktop/Computer Science"
    "/CALPOLY/Spring Quarter/CSC321/Facial-Recognition/config.yaml"
)


class TestWriteRunManifest:
    def _load_cfg(self):
        from src.utils.config_loader import ConfigLoader
        return ConfigLoader().load(_CONFIG_YAML)

    def test_creates_run_manifest_json(self, manager: ArtifactManager, tmp: Path):
        cfg = self._load_cfg()
        out_dir = tmp / "artifacts"
        manager.write_run_manifest(out_dir, cfg)
        assert (out_dir / "run_manifest.json").exists(), "run_manifest.json must be created"

    def test_manifest_contains_all_config_fields(self, manager: ArtifactManager, tmp: Path):
        import dataclasses
        from src.utils.config_loader import Config

        cfg = self._load_cfg()
        out_dir = tmp / "artifacts2"
        manager.write_run_manifest(out_dir, cfg)

        with (out_dir / "run_manifest.json").open() as fh:
            stored = json.load(fh)

        expected_fields = {f.name for f in dataclasses.fields(Config)}
        missing = expected_fields - set(stored.keys())
        assert not missing, f"run_manifest.json missing fields: {missing}"

    def test_creates_directory_if_not_exists(self, manager: ArtifactManager, tmp: Path):
        cfg = self._load_cfg()
        out_dir = tmp / "deep" / "nested" / "dir"
        assert not out_dir.exists()
        manager.write_run_manifest(out_dir, cfg)
        assert (out_dir / "run_manifest.json").exists()

    def test_manifest_values_match_config(self, manager: ArtifactManager, tmp: Path):
        cfg = self._load_cfg()
        out_dir = tmp / "arts"
        manager.write_run_manifest(out_dir, cfg)

        with (out_dir / "run_manifest.json").open() as fh:
            stored = json.load(fh)

        assert stored["random_seed"] == cfg.random_seed
        assert stored["dataset_type"] == cfg.dataset_type
        assert stored["epochs"] == cfg.epochs
