"""
tests/unit/test_face_detector.py

Unit tests for src/preprocessing/face_detector.py.

Covers:
  - DetectionResult dataclass construction
  - FaceDetector._select_best (core selection logic, R2.1)
  - make_detector factory with an invalid backend string (R2.7)
  - HaarFaceDetector happy-path on a synthetic image (smoke test)
  - MTCNNFaceDetector / RetinaFaceFaceDetector raise DetectorInitError
    when the optional library is absent (R2.7)
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.preprocessing.face_detector import (
    DetectionResult,
    FaceDetector,
    HaarFaceDetector,
    make_detector,
)
from src.utils.exceptions import DetectorInitError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(backend: str = "haar") -> MagicMock:
    """Return a minimal mock Config with a given detector_backend."""
    cfg = MagicMock()
    cfg.detector_backend = backend
    return cfg


def _synthetic_bgr(height: int = 100, width: int = 100) -> np.ndarray:
    """Return a solid-grey BGR image."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_no_face(self):
        r = DetectionResult(bbox=None, confidence=0.0)
        assert r.bbox is None
        assert r.confidence == 0.0

    def test_with_face(self):
        r = DetectionResult(bbox=(10, 20, 50, 60), confidence=0.95)
        assert r.bbox == (10, 20, 50, 60)
        assert r.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# FaceDetector._select_best  (R2.1)
# ---------------------------------------------------------------------------


class TestSelectBest:
    """Tests for the static helper that implements R2.1."""

    def test_empty_candidates_returns_no_face(self):
        result = FaceDetector._select_best([])
        assert result.bbox is None
        assert result.confidence == 0.0

    def test_single_candidate_is_returned(self):
        candidates = [((5, 10, 40, 50), 0.8)]
        result = FaceDetector._select_best(candidates)
        assert result.bbox == (5, 10, 40, 50)
        assert result.confidence == pytest.approx(0.8)

    def test_highest_confidence_selected(self):
        candidates = [
            ((0, 0, 10, 10), 0.5),
            ((0, 0, 10, 10), 0.9),
            ((0, 0, 10, 10), 0.3),
        ]
        result = FaceDetector._select_best(candidates)
        assert result.confidence == pytest.approx(0.9)

    def test_tie_broken_by_largest_area(self):
        """When two detections share the same confidence, pick the larger bbox."""
        candidates = [
            ((0, 0, 30, 30), 0.7),  # area = 900
            ((0, 0, 50, 50), 0.7),  # area = 2500  ← should win
            ((0, 0, 20, 20), 0.7),  # area = 400
        ]
        result = FaceDetector._select_best(candidates)
        assert result.bbox == (0, 0, 50, 50)
        assert result.confidence == pytest.approx(0.7)

    def test_lower_confidence_larger_area_not_selected(self):
        """A larger bounding box with lower confidence must NOT win."""
        candidates = [
            ((0, 0, 100, 100), 0.6),  # large but lower confidence
            ((0, 0, 20, 20),   0.9),  # small but higher confidence ← should win
        ]
        result = FaceDetector._select_best(candidates)
        assert result.bbox == (0, 0, 20, 20)
        assert result.confidence == pytest.approx(0.9)

    def test_multiple_ties_largest_wins(self):
        """Three-way confidence tie — the biggest area wins."""
        candidates = [
            ((0, 0, 10, 20), 1.0),  # area = 200
            ((0, 0, 30, 40), 1.0),  # area = 1200  ← should win
            ((0, 0, 15, 15), 1.0),  # area = 225
        ]
        result = FaceDetector._select_best(candidates)
        assert result.bbox == (0, 0, 30, 40)


# ---------------------------------------------------------------------------
# make_detector factory  (R2.2, R2.7)
# ---------------------------------------------------------------------------


class TestMakeDetector:
    def test_unknown_backend_raises_detector_init_error(self):
        cfg = _make_config("unknown_backend")
        with pytest.raises(DetectorInitError, match="unrecognised detector_backend"):
            make_detector(cfg)

    def test_error_message_lists_accepted_values(self):
        cfg = _make_config("bad")
        with pytest.raises(DetectorInitError) as exc_info:
            make_detector(cfg)
        msg = str(exc_info.value)
        for backend in ("haar", "dnn", "mtcnn", "retinaface"):
            assert backend in msg

    def test_haar_backend_returns_haar_detector(self):
        """HaarFaceDetector can be constructed without extra optional packages."""
        cfg = _make_config("haar")
        detector = make_detector(cfg)
        assert isinstance(detector, HaarFaceDetector)

    def test_backend_name_case_insensitive(self):
        """Backend string matching should be case-insensitive."""
        cfg = _make_config("HAAR")
        detector = make_detector(cfg)
        assert isinstance(detector, HaarFaceDetector)


# ---------------------------------------------------------------------------
# HaarFaceDetector smoke test
# ---------------------------------------------------------------------------


class TestHaarFaceDetector:
    def test_detect_returns_detection_result(self):
        """detect() must always return a DetectionResult (bbox may be None)."""
        detector = HaarFaceDetector()
        img = _synthetic_bgr()
        result = detector.detect(img)
        assert isinstance(result, DetectionResult)

    def test_detect_on_grayscale_image(self):
        """detect() must handle a single-channel grayscale input."""
        detector = HaarFaceDetector()
        gray = np.full((100, 100), 128, dtype=np.uint8)
        result = detector.detect(gray)
        assert isinstance(result, DetectionResult)

    def test_no_face_in_blank_image(self):
        """A solid-colour image contains no face; bbox should be None."""
        detector = HaarFaceDetector()
        img = _synthetic_bgr()
        result = detector.detect(img)
        # A blank image should not produce a detection.
        assert result.bbox is None


# ---------------------------------------------------------------------------
# Optional-library guards  (R2.7)
# ---------------------------------------------------------------------------


class TestMTCNNDetectorImportGuard:
    def test_raises_when_facenet_pytorch_missing(self):
        """MTCNNFaceDetector must raise DetectorInitError when facenet_pytorch
        is not installed."""
        from src.preprocessing.face_detector import MTCNNFaceDetector

        # Temporarily hide the facenet_pytorch module.
        with patch.dict(sys.modules, {"facenet_pytorch": None}):
            with pytest.raises(DetectorInitError, match="facenet-pytorch"):
                MTCNNFaceDetector()


class TestRetinaFaceDetectorImportGuard:
    def test_raises_when_retinaface_missing(self):
        """RetinaFaceFaceDetector must raise DetectorInitError when retinaface
        is not installed."""
        from src.preprocessing.face_detector import RetinaFaceFaceDetector

        with patch.dict(sys.modules, {"retinaface": None}):
            with pytest.raises(DetectorInitError, match="retinaface"):
                RetinaFaceFaceDetector()


# ---------------------------------------------------------------------------
# DNNFaceDetector init guard  (R2.7)
# ---------------------------------------------------------------------------


class TestDNNDetectorInitGuard:
    def test_raises_when_model_files_missing(self, tmp_path, monkeypatch):
        """DNNFaceDetector must raise DetectorInitError when weight files are
        absent from the 'models/' directory."""
        import src.preprocessing.face_detector as fd_module

        # Point _DNN_MODEL_DIR at an empty tmp directory.
        monkeypatch.setattr(fd_module, "_DNN_MODEL_DIR", tmp_path)
        from src.preprocessing.face_detector import DNNFaceDetector

        with pytest.raises(DetectorInitError, match="required model file"):
            DNNFaceDetector()

    def test_error_message_contains_download_hint(self, tmp_path, monkeypatch):
        import src.preprocessing.face_detector as fd_module

        monkeypatch.setattr(fd_module, "_DNN_MODEL_DIR", tmp_path)
        from src.preprocessing.face_detector import DNNFaceDetector

        with pytest.raises(DetectorInitError) as exc_info:
            DNNFaceDetector()
        assert "Download" in str(exc_info.value) or "download" in str(exc_info.value)
