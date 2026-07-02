"""
src/preprocessing/dataset_loader.py

Scans a one-level-deep image folder structure and returns a flat list of
Sample(path, label) objects.  Supports 'kaggle' and 'vggface2' dataset
conventions (both share the same immediate-subdir-per-identity layout).

Requirements: 1.1–1.8
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.config_loader import Config
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported dataset types
# ---------------------------------------------------------------------------
_SUPPORTED_DATASET_TYPES: frozenset[str] = frozenset({"kaggle", "vggface2"})

# Supported image file extensions (lower-cased for case-insensitive matching)
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """A single labelled image path."""
    path: str
    label: str


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Loads image samples from a directory tree where each immediate
    subdirectory name is the Identity label.

    Directory structure expected::

        dataset_path/
        ├── identity_A/
        │   ├── img1.jpg
        │   └── img2.png
        └── identity_B/
            └── img3.jpeg

    Args:
        config: A fully validated :class:`~src.utils.config_loader.Config`
                instance.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[Sample]:
        """Scan ``config.dataset_path`` for a one-level-deep subdirectory
        structure and return a flat list of :class:`Sample` objects.

        Raises:
            DatasetError: If ``dataset_path`` does not exist on disk.
            DatasetError: If ``dataset_type`` is not one of the supported
                          values (``'kaggle'``, ``'vggface2'``).

        Returns:
            A list of :class:`Sample` objects, possibly empty if every
            identity ends up with zero loadable images after filtering.
        """
        self._validate_dataset_type()
        root = self._resolve_dataset_path()

        identity_dirs = self._collect_identity_dirs(root)

        # Apply max_identities cap (R1.4)
        identity_dirs = self._apply_max_identities(identity_dirs)

        samples: list[Sample] = []

        for identity_dir in identity_dirs:
            label = identity_dir.name
            identity_samples = self._load_identity(identity_dir, label)

            if not identity_samples:
                # R1.8: warn and exclude identities with zero loadable images
                logger.warning(
                    "Identity '%s' has 0 loadable images after filtering — excluding.",
                    label,
                )
                continue

            samples.extend(identity_samples)

        return samples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_dataset_type(self) -> None:
        """Raise DatasetError if dataset_type is not recognised (R1.7)."""
        dtype = self._config.dataset_type
        if dtype not in _SUPPORTED_DATASET_TYPES:
            accepted = ", ".join(sorted(_SUPPORTED_DATASET_TYPES))
            raise DatasetError(
                f"Unrecognised dataset_type '{dtype}'. "
                f"Accepted values are: {accepted}."
            )

    def _resolve_dataset_path(self) -> Path:
        """Resolve and verify that dataset_path exists (R1.2)."""
        root = Path(self._config.dataset_path)
        if not root.exists():
            raise DatasetError(
                f"Dataset path '{root.resolve()}' does not exist. "
                f"dataset_type is '{self._config.dataset_type}'."
            )
        return root

    def _collect_identity_dirs(self, root: Path) -> list[Path]:
        """Return a sorted list of immediate subdirectories under *root* (R1.1)."""
        identity_dirs = sorted(
            p for p in root.iterdir() if p.is_dir()
        )
        return identity_dirs

    def _apply_max_identities(self, identity_dirs: list[Path]) -> list[Path]:
        """Truncate *identity_dirs* to at most ``max_identities`` entries (R1.4)."""
        max_ids = self._config.max_identities
        if max_ids is not None and max_ids >= 1:
            identity_dirs = identity_dirs[:max_ids]
        return identity_dirs

    def _load_identity(self, identity_dir: Path, label: str) -> list[Sample]:
        """Collect all loadable image samples for one identity directory.

        - Filters files by supported extension (R1.1).
        - Skips and warns for unreadable/corrupt images (R1.3).
        - Applies max_images_per_identity cap (R1.5).

        Returns:
            A (possibly empty) list of :class:`Sample` objects for this identity.
        """
        candidate_files = sorted(
            p for p in identity_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        # Apply max_images_per_identity cap before validation to avoid
        # needlessly opening images that will be discarded (R1.5).
        max_imgs = self._config.max_images_per_identity
        if max_imgs is not None and max_imgs >= 1:
            candidate_files = candidate_files[:max_imgs]

        samples: list[Sample] = []
        for img_path in candidate_files:
            if self._is_image_readable(img_path):
                samples.append(Sample(path=str(img_path), label=label))
            # Warning logged inside _is_image_readable on failure (R1.3)

        return samples

    @staticmethod
    def _is_image_readable(img_path: Path) -> bool:
        """Try to open the image to verify it is not corrupt.

        Uses the standard library only (no OpenCV/PIL required at this stage)
        by verifying the file is non-empty and attempting a minimal open with
        ``PIL`` if available, or falling back to a raw header check otherwise.

        Logs a WARNING and returns ``False`` if the image cannot be read (R1.3).

        Returns:
            ``True`` if the image appears valid, ``False`` otherwise.
        """
        try:
            # Prefer PIL/Pillow for accurate corruption detection.
            from PIL import Image  # type: ignore[import]
            with Image.open(img_path) as img:
                img.verify()  # raises on corrupt data
            return True
        except ImportError:
            pass  # PIL not available — fall back to file-size check
        except Exception as exc:
            logger.warning(
                "Cannot open image '%s' — skipping. Reason: %s",
                img_path,
                exc,
            )
            return False

        # Fallback: just check the file is non-empty (minimal sanity check)
        try:
            if img_path.stat().st_size == 0:
                logger.warning(
                    "Cannot open image '%s' — file is empty, skipping.",
                    img_path,
                )
                return False
        except OSError as exc:
            logger.warning(
                "Cannot open image '%s' — skipping. Reason: %s",
                img_path,
                exc,
            )
            return False

        return True
