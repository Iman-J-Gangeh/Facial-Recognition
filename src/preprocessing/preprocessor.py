"""
src/preprocessing/preprocessor.py

Shared image preprocessor used by both the classical and deep learning pipelines.
Handles face detection, cropping, resizing, grayscale conversion, and fallbacks.

Requirements: 2.3–2.6, 2.8, 17.1–17.3, 17.6
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.preprocessing.dataset_loader import Sample
from src.preprocessing.face_detector import FaceDetector
from src.utils.exceptions import ConfigError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config_loader import Config

logger = get_logger(__name__)

_VALID_FALLBACKS = frozenset({"skip", "use_full"})


@dataclass
class ProcessedSample:
    """A preprocessed image ready for feature extraction or model input."""
    image: np.ndarray   # grayscale uint8 for classical, float32 RGB for deep
    label: str
    source_path: str


class Preprocessor:
    """Detects, crops, resizes, and formats face images for both pipelines.

    Args:
        config: Validated Config instance.
        detector: An initialised FaceDetector backend.

    Raises:
        ConfigError: If config.no_face_fallback is not 'skip' or 'use_full'.
    """

    def __init__(self, config: "Config", detector: FaceDetector) -> None:
        if config.no_face_fallback not in _VALID_FALLBACKS:
            raise ConfigError(
                f"Invalid no_face_fallback value '{config.no_face_fallback}'. "
                f"Accepted values are: {sorted(_VALID_FALLBACKS)}."
            )
        self._config = config
        self._detector = detector
        self._w, self._h = config.image_size  # (width, height)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_for_classical(self, sample: Sample) -> ProcessedSample | None:
        """Detect face, crop, resize, convert to grayscale.

        Returns None if face detection fails and fallback is 'skip'.
        Logs appropriate warnings per R17.2/17.3.
        """
        bgr = self._load_image(sample.path)
        if bgr is None:
            return self._apply_fallback_on_no_image(sample)

        crop = self._detect_and_crop(bgr, sample.path)
        if crop is None:
            return None  # skip already logged

        resized = cv2.resize(crop, (self._w, self._h))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        return ProcessedSample(image=gray, label=sample.label, source_path=sample.path)

    def process_for_deep(self, sample: Sample) -> ProcessedSample | None:
        """Detect face, crop, resize, return float32 RGB array.

        Normalization is applied later inside the DataLoader transform.
        """
        bgr = self._load_image(sample.path)
        if bgr is None:
            return self._apply_fallback_on_no_image(sample)

        crop = self._detect_and_crop(bgr, sample.path)
        if crop is None:
            return None

        resized = cv2.resize(crop, (self._w, self._h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ProcessedSample(image=rgb, label=sample.label, source_path=sample.path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray | None:
        """Load a BGR image from disk. Returns None and logs WARNING on failure (R2.8, R17.1)."""
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("cv2.imread returned None")
            return img
        except Exception as exc:
            logger.warning("Cannot decode image '%s' — %s", path, exc)
            return None

    def _detect_and_crop(self, bgr: np.ndarray, path: str) -> np.ndarray | None:
        """Run face detection. Returns cropped BGR region or None if no face + skip fallback."""
        result = self._detector.detect(bgr)
        if result.bbox is None:
            fallback = self._config.no_face_fallback
            if fallback == "skip":
                logger.warning("no face detected — image skipped: %s", path)
                return None
            else:  # "use_full"
                logger.warning("no face detected — using full image: %s", path)
                return bgr

        x, y, w, h = result.bbox
        # Clamp to image bounds
        ih, iw = bgr.shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(w, iw - x)
        h = min(h, ih - y)
        if w <= 0 or h <= 0:
            logger.warning("no face detected — image skipped (invalid bbox): %s", path)
            if self._config.no_face_fallback == "use_full":
                return bgr
            return None
        return bgr[y:y + h, x:x + w]

    def _apply_fallback_on_no_image(self, sample: Sample) -> ProcessedSample | None:
        """Handle the case where the image could not be loaded at all."""
        fallback = self._config.no_face_fallback
        if fallback == "skip":
            logger.warning("no face detected — image skipped: %s", sample.path)
            return None
        # use_full: return a black image of the configured size
        logger.warning("no face detected — using full image: %s", sample.path)
        blank = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        return ProcessedSample(image=blank, label=sample.label, source_path=sample.path)
