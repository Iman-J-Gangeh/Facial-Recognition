"""
src/utils/exceptions.py

All custom exception classes used throughout the facial recognition
comparison system.  Importing from this single module avoids circular
dependencies and provides a single place to find every domain error.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Raised for any configuration problem: missing file, missing required
    key, bad type coercion, invalid value, etc.

    This duplicates the definition in ``config_loader.py`` for modules that
    want to catch the error without importing the entire config_loader.  Both
    names resolve to the same class via the ``src.utils.exceptions`` namespace.
    """


class DatasetError(Exception):
    """Raised when the dataset path does not exist, the dataset type is
    unrecognised, or the dataset contains no valid images after filtering."""


class DetectorInitError(Exception):
    """Raised when a face-detection backend cannot be initialised, e.g. due
    to missing model weights or an unsupported library version."""

class PreprocessingError(Exception):
    """Raised when image preprocessing fails, such as face detection,
    cropping, resizing, or feature-preparation errors."""
    
class ArtifactNotFoundError(FileNotFoundError):
    """Raised when an artifact file expected on disk is absent.

    Inherits from :class:`FileNotFoundError` so callers can catch either the
    domain-specific or the built-in variant.
    """


class ArtifactError(Exception):
    """Raised for artifact-related failures other than a missing file:
    corrupt data, incompatible format, failed save, etc."""


class SIFTExtractionError(Exception):
    """Raised when SIFT descriptor extraction produces no usable descriptors
    from the entire training split."""


class VocabularyError(Exception):
    """Raised when visual-vocabulary building fails, e.g. because the
    descriptor collection passed to K-means is empty."""


class VerificationError(Exception):
    """Raised when the classical verification engine cannot produce a result,
    e.g. because both input image sets are empty or all preprocessing failed."""


class ModelConfigError(Exception):
    """Raised when the deep learning model configuration is invalid:
    unrecognised architecture string, ambiguous value, etc."""
