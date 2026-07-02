"""
src/utils/artifact_manager.py

Centralised artifact persistence utilities used by every pipeline component
that saves or loads objects to/from disk.

Responsibilities
----------------
* ``save``              — Serialize an object and write a companion JSON manifest.
* ``load``              — Deserialize an artifact; warn on config drift.
* ``measure_size_mb``   — Sum file sizes in megabytes.
* ``write_run_manifest``— Capture the full ``Config`` to ``run_manifest.json``.

Requirements: 16.3–16.6
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
from pathlib import Path
from typing import Any

from src.utils.exceptions import ArtifactError, ArtifactNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Extensions that trigger torch.save / torch.load instead of pickle.
_TORCH_EXTENSIONS: frozenset[str] = frozenset({".pt", ".pth"})


def _is_torch_path(path: Path) -> bool:
    return path.suffix.lower() in _TORCH_EXTENSIONS


def _manifest_path(artifact_path: Path) -> Path:
    """Return the companion manifest path for *artifact_path*.

    For ``kmeans_vocab.pkl`` → ``kmeans_vocab.pkl.manifest.json``
    """
    return artifact_path.parent / (artifact_path.name + ".manifest.json")


class ArtifactManager:
    """Manages serialization, deserialization, size measurement, and run
    manifests for all pipeline artifacts.

    All methods are instance methods (no config required at construction
    time) so a single shared instance can serve the entire process.
    """

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(self, obj: Any, path: str | Path, metadata: dict) -> None:
        """Serialize *obj* to *path* and write a companion manifest.

        Serialization strategy:

        * ``.pt`` / ``.pth`` extensions → :func:`torch.save`
        * Everything else              → :mod:`pickle`

        A companion file ``<artifact>.manifest.json`` is written alongside
        the artifact containing *metadata* as a JSON object.

        Args:
            obj:      The Python object to persist.
            path:     Destination file path (string or :class:`pathlib.Path`).
            metadata: Arbitrary key/value dict saved verbatim to the manifest.

        Raises:
            ArtifactError: If the save or manifest write fails.
        """
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if _is_torch_path(artifact_path):
                import torch  # local import — optional dependency
                torch.save(obj, artifact_path)
                logger.debug("Saved torch artifact → %s", artifact_path)
            else:
                with artifact_path.open("wb") as fh:
                    pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug("Saved pickle artifact → %s", artifact_path)
        except Exception as exc:
            raise ArtifactError(
                f"Failed to save artifact to '{artifact_path}': {exc}"
            ) from exc

        # Write the companion manifest.
        manifest = _manifest_path(artifact_path)
        try:
            with manifest.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, default=str)
            logger.debug("Wrote manifest → %s", manifest)
        except Exception as exc:
            raise ArtifactError(
                f"Artifact saved but manifest write failed at '{manifest}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def load(self, path: str | Path, expected_meta: dict | None = None) -> Any:
        """Deserialize and return the artifact at *path*.

        Metadata compatibility check (Requirement 16.4 / 16.5):

        If a companion manifest exists and *expected_meta* is provided,
        each field in *expected_meta* is compared against the recorded
        value.  A WARNING is logged for every differing field but loading
        continues regardless (the artifact is never rejected).

        Args:
            path:          Path to the artifact file.
            expected_meta: Dict of field → expected-value pairs to check
                           against the stored manifest.  ``None`` skips
                           the check entirely.

        Returns:
            The deserialized Python object.

        Raises:
            ArtifactNotFoundError: If *path* does not exist on disk.
            ArtifactError:         If deserialization fails (corrupt file,
                                   incompatible format, etc.).
        """
        artifact_path = Path(path)

        if not artifact_path.exists():
            raise ArtifactNotFoundError(
                f"Artifact not found: '{artifact_path}'. "
                "Run the appropriate training script to generate it."
            )

        # --- Metadata compatibility check --------------------------------
        if expected_meta:
            manifest = _manifest_path(artifact_path)
            if manifest.exists():
                try:
                    with manifest.open("r", encoding="utf-8") as fh:
                        recorded: dict = json.load(fh)
                    for field, expected_val in expected_meta.items():
                        recorded_val = recorded.get(field)
                        if recorded_val != expected_val:
                            logger.warning(
                                "Artifact metadata mismatch for field '%s': "
                                "recorded=%r, expected=%r — loading anyway.",
                                field,
                                recorded_val,
                                expected_val,
                            )
                except Exception as exc:
                    logger.warning(
                        "Could not read manifest '%s': %s — skipping compatibility check.",
                        manifest,
                        exc,
                    )
            else:
                logger.warning(
                    "No manifest found at '%s' — skipping compatibility check.",
                    manifest,
                )

        # --- Deserialize -------------------------------------------------
        try:
            if _is_torch_path(artifact_path):
                import torch  # local import — optional dependency
                obj = torch.load(artifact_path, map_location="cpu", weights_only=False)
                logger.debug("Loaded torch artifact ← %s", artifact_path)
            else:
                with artifact_path.open("rb") as fh:
                    obj = pickle.load(fh)
                logger.debug("Loaded pickle artifact ← %s", artifact_path)
        except Exception as exc:
            raise ArtifactError(
                f"Failed to deserialize artifact at '{artifact_path}': {exc}"
            ) from exc

        return obj

    # ------------------------------------------------------------------
    # measure_size_mb
    # ------------------------------------------------------------------

    def measure_size_mb(self, paths: list[str | Path]) -> float:
        """Return the total size in megabytes of the files in *paths*.

        Files that do not exist are silently skipped (they contribute 0
        bytes, and a DEBUG message is logged).

        Args:
            paths: Iterable of file paths to measure.

        Returns:
            Combined size in megabytes as a :class:`float`.
        """
        total_bytes: int = 0
        for p in paths:
            fp = Path(p)
            if fp.exists():
                total_bytes += fp.stat().st_size
            else:
                logger.debug(
                    "measure_size_mb: '%s' not found — skipping.", fp
                )
        return total_bytes / (1024 * 1024)

    # ------------------------------------------------------------------
    # write_run_manifest
    # ------------------------------------------------------------------

    def write_run_manifest(self, artifact_dir: str | Path, config: Any) -> None:
        """Serialize the full ``Config`` to ``run_manifest.json`` inside
        *artifact_dir*.

        All :class:`~src.utils.config_loader.Config` dataclass fields are
        recorded, with non-JSON-serializable types (tuples, etc.) converted
        to their closest JSON equivalent via the ``default=str`` fallback.

        Args:
            artifact_dir: Directory in which to write ``run_manifest.json``.
            config:       A :class:`~src.utils.config_loader.Config` instance
                          (or any dataclass).

        Raises:
            ArtifactError: If the directory cannot be created or the file
                           cannot be written.
        """
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_file = out_dir / "run_manifest.json"

        # Convert dataclass → plain dict, then serialise.
        try:
            if dataclasses.is_dataclass(config) and not isinstance(config, type):
                config_dict = dataclasses.asdict(config)
            elif hasattr(config, "__dict__"):
                config_dict = dict(vars(config))
            else:
                config_dict = dict(config)  # type: ignore[arg-type]
        except Exception as exc:
            raise ArtifactError(
                f"Cannot convert config to dict for run manifest: {exc}"
            ) from exc

        try:
            with manifest_file.open("w", encoding="utf-8") as fh:
                json.dump(config_dict, fh, indent=2, default=str)
            logger.info("Run manifest written → %s", manifest_file)
        except Exception as exc:
            raise ArtifactError(
                f"Failed to write run manifest to '{manifest_file}': {exc}"
            ) from exc
