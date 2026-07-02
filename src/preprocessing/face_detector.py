"""
src/preprocessing/face_detector.py

FaceDetector abstract base class, four concrete backend implementations,
and a factory function (make_detector).

Face-selection rule (R2.1):
  When multiple faces are detected, select the one with the highest
  confidence.  On a confidence tie, select the one with the largest
  bounding-box pixel area (w * h).

Supported backends (R2.2):
  "haar"        – OpenCV Haar cascade (cv2.CascadeClassifier)
  "dnn"         – OpenCV DNN (res10_300x300_ssd_iter_140000)
  "mtcnn"       – facenet_pytorch.MTCNN
  "retinaface"  – retinaface.RetinaFace

Initialization failures (R2.7):
  make_detector raises DetectorInitError if a backend cannot be started
  (missing weights file, missing optional library, etc.).

Requirements: 2.1, 2.2, 2.7
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.utils.exceptions import DetectorInitError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config_loader import Config

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Result of a single face-detection call.

    Attributes:
        bbox: ``(x, y, w, h)`` bounding box of the best detected face, or
              ``None`` when no face was found.
        confidence: Detection confidence in ``[0.0, 1.0]``.  ``0.0`` when no
                    face was detected.
    """

    bbox: tuple[int, int, int, int] | None
    confidence: float


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class FaceDetector(ABC):
    """Abstract face detector.  All concrete backends inherit from this."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect the best face in *image*.

        Args:
            image: An ``np.ndarray`` in BGR (OpenCV) colour order with dtype
                   ``uint8``.  Shape must be ``(H, W, 3)`` or ``(H, W)``.

        Returns:
            A :class:`DetectionResult` whose ``bbox`` is ``None`` when no
            face was found.
        """

    # ------------------------------------------------------------------
    # Shared helper: pick the best detection among many candidates
    # ------------------------------------------------------------------

    @staticmethod
    def _select_best(
        candidates: list[tuple[tuple[int, int, int, int], float]],
    ) -> DetectionResult:
        """Select the best detection from a list of ``(bbox, confidence)`` pairs.

        Selection rule (R2.1):
          1. Pick the candidate(s) with the **highest confidence**.
          2. Among ties, pick the one with the **largest bounding-box area**
             (``w * h``).

        Returns a :class:`DetectionResult` with ``bbox=None`` and
        ``confidence=0.0`` when *candidates* is empty.
        """
        if not candidates:
            return DetectionResult(bbox=None, confidence=0.0)

        # Step 1 — find the maximum confidence value.
        max_conf = max(conf for _, conf in candidates)

        # Step 2 — collect all candidates that share the maximum confidence.
        top_candidates = [(bbox, conf) for bbox, conf in candidates if conf == max_conf]

        # Step 3 — among ties, pick the largest area.
        best_bbox, best_conf = max(top_candidates, key=lambda item: item[0][2] * item[0][3])
        return DetectionResult(bbox=best_bbox, confidence=best_conf)


# ---------------------------------------------------------------------------
# HaarFaceDetector
# ---------------------------------------------------------------------------


class HaarFaceDetector(FaceDetector):
    """OpenCV Haar cascade face detector.

    Uses ``haarcascade_frontalface_default.xml`` which ships with OpenCV.
    The cascade is loaded once during ``__init__``; ``DetectorInitError`` is
    raised if OpenCV cannot locate or load the cascade file.
    """

    _CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

    def __init__(self) -> None:
        cascade_path = self._find_cascade()
        self._classifier = cv2.CascadeClassifier(cascade_path)
        if self._classifier.empty():
            raise DetectorInitError(
                f"HaarFaceDetector: failed to load cascade from '{cascade_path}'. "
                "Ensure opencv-python is installed correctly and the data files are present."
            )
        logger.debug("HaarFaceDetector initialised (cascade: %s)", cascade_path)

    @classmethod
    def _find_cascade(cls) -> str:
        """Locate the Haar cascade XML file bundled with OpenCV."""
        # cv2.data.haarcascades is available in opencv-python >= 3.x
        cv2_data_path = getattr(cv2, "data", None)
        if cv2_data_path is not None:
            candidate = os.path.join(cv2_data_path.haarcascades, cls._CASCADE_FILENAME)
            if os.path.isfile(candidate):
                return candidate

        # Fall back: search site-packages
        import importlib.util

        spec = importlib.util.find_spec("cv2")
        if spec and spec.origin:
            cv2_dir = Path(spec.origin).parent
            for root, _dirs, files in os.walk(str(cv2_dir)):
                if cls._CASCADE_FILENAME in files:
                    return os.path.join(root, cls._CASCADE_FILENAME)

        raise DetectorInitError(
            f"HaarFaceDetector: cannot locate '{cls._CASCADE_FILENAME}' in the "
            "OpenCV data directory.  Please reinstall opencv-python."
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect faces using the Haar cascade."""
        gray = _to_gray(image)
        detections = self._classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if detections is None or len(detections) == 0:
            return DetectionResult(bbox=None, confidence=0.0)

        # Haar cascade does not return per-detection confidences.
        # Assign uniform confidence = 1.0; area tie-breaking still applies.
        candidates = [
            ((int(x), int(y), int(w), int(h)), 1.0)
            for x, y, w, h in detections
        ]
        return self._select_best(candidates)


# ---------------------------------------------------------------------------
# DNNFaceDetector
# ---------------------------------------------------------------------------

_DNN_MODEL_DIR = Path("models")
_DNN_PROTO = "deploy.prototxt"
_DNN_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"
_DNN_CONFIDENCE_THRESHOLD = 0.5


class DNNFaceDetector(FaceDetector):
    """OpenCV DNN face detector.

    Uses ``res10_300x300_ssd_iter_140000.caffemodel`` + ``deploy.prototxt``.
    Both files must be present in the ``models/`` directory at the project
    root.  If either is missing, :exc:`DetectorInitError` is raised with a
    download hint.
    """

    def __init__(self) -> None:
        proto_path = _DNN_MODEL_DIR / _DNN_PROTO
        weights_path = _DNN_MODEL_DIR / _DNN_WEIGHTS

        missing = [str(p) for p in (proto_path, weights_path) if not p.exists()]
        if missing:
            raise DetectorInitError(
                f"DNNFaceDetector: required model file(s) not found: {missing}. "
                "Download them from the OpenCV GitHub repository:\n"
                "  deploy.prototxt: https://github.com/opencv/opencv/blob/master/"
                "samples/dnn/face_detector/deploy.prototxt\n"
                "  caffemodel: https://github.com/opencv/opencv_3rdparty/raw/"
                "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel\n"
                f"Place both files in '{_DNN_MODEL_DIR.resolve()}'."
            )

        try:
            self._net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
        except cv2.error as exc:
            raise DetectorInitError(
                f"DNNFaceDetector: OpenCV failed to load the model — {exc}. "
                "The model files in 'models/' may be corrupt; re-download them."
            ) from exc

        logger.debug("DNNFaceDetector initialised (weights: %s)", weights_path)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect faces using the OpenCV DNN model."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        detections = self._net.forward()  # shape: (1, 1, N, 7)

        candidates: list[tuple[tuple[int, int, int, int], float]] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < _DNN_CONFIDENCE_THRESHOLD:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                candidates.append(((x1, y1, bw, bh), confidence))

        return self._select_best(candidates)


# ---------------------------------------------------------------------------
# MTCNNFaceDetector
# ---------------------------------------------------------------------------


class MTCNNFaceDetector(FaceDetector):
    """MTCNN face detector via ``facenet_pytorch``.

    The ``facenet-pytorch`` package is an optional dependency.  If it is
    not installed, :exc:`DetectorInitError` is raised at construction time.
    """

    def __init__(self) -> None:
        try:
            from facenet_pytorch import MTCNN  # type: ignore[import]
        except ImportError as exc:
            raise DetectorInitError(
                "MTCNNFaceDetector: 'facenet-pytorch' is not installed. "
                "Install it with: pip install facenet-pytorch"
            ) from exc

        try:
            # keep_all=True so we can pick the best face ourselves.
            self._mtcnn = MTCNN(keep_all=True, post_process=False)
        except Exception as exc:  # noqa: BLE001
            raise DetectorInitError(
                f"MTCNNFaceDetector: failed to initialise MTCNN — {exc}"
            ) from exc

        logger.debug("MTCNNFaceDetector initialised")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect faces using MTCNN."""
        from PIL import Image as PILImage  # type: ignore[import]

        # MTCNN expects a PIL Image or RGB numpy array.
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_img = PILImage.fromarray(rgb)

        boxes, probs = self._mtcnn.detect(pil_img)

        if boxes is None or len(boxes) == 0:
            return DetectionResult(bbox=None, confidence=0.0)

        h, w = image.shape[:2]
        candidates: list[tuple[tuple[int, int, int, int], float]] = []
        for box, prob in zip(boxes, probs):
            if prob is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                candidates.append(((x1, y1, bw, bh), float(prob)))

        return self._select_best(candidates)


# ---------------------------------------------------------------------------
# RetinaFaceFaceDetector
# ---------------------------------------------------------------------------


class RetinaFaceFaceDetector(FaceDetector):
    """RetinaFace face detector via the ``retinaface`` package.

    The ``retinaface`` package is an optional dependency.  If it is not
    installed, :exc:`DetectorInitError` is raised at construction time.
    """

    def __init__(self) -> None:
        try:
            import retinaface as _retinaface  # type: ignore[import]  # noqa: F401
            from retinaface import RetinaFace  # type: ignore[import]

            self._RetinaFace = RetinaFace
        except ImportError as exc:
            raise DetectorInitError(
                "RetinaFaceFaceDetector: 'retinaface' is not installed. "
                "Install it with: pip install retina-face"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise DetectorInitError(
                f"RetinaFaceFaceDetector: failed to initialise RetinaFace — {exc}"
            ) from exc

        logger.debug("RetinaFaceFaceDetector initialised")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect faces using RetinaFace."""
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image

        try:
            results = self._RetinaFace.detect_faces(bgr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RetinaFaceFaceDetector.detect: unexpected error — %s", exc)
            return DetectionResult(bbox=None, confidence=0.0)

        if not results or not isinstance(results, dict):
            return DetectionResult(bbox=None, confidence=0.0)

        h, w = image.shape[:2]
        candidates: list[tuple[tuple[int, int, int, int], float]] = []
        for face_data in results.values():
            score = float(face_data.get("score", 0.0))
            facial_area = face_data.get("facial_area")
            if facial_area is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in facial_area)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                candidates.append(((x1, y1, bw, bh), score))

        return self._select_best(candidates)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKEND_MAP: dict[str, type[FaceDetector]] = {
    "haar": HaarFaceDetector,
    "dnn": DNNFaceDetector,
    "mtcnn": MTCNNFaceDetector,
    "retinaface": RetinaFaceFaceDetector,
}


def make_detector(config: "Config") -> FaceDetector:
    """Instantiate and return the :class:`FaceDetector` specified by *config*.

    Args:
        config: A fully validated :class:`~src.utils.config_loader.Config`
                instance.  ``config.detector_backend`` must be one of
                ``"haar"``, ``"dnn"``, ``"mtcnn"``, or ``"retinaface"``.

    Returns:
        A ready-to-use :class:`FaceDetector` instance.

    Raises:
        :exc:`~src.utils.exceptions.DetectorInitError`: If the requested
            backend string is unrecognised OR if the backend cannot be
            initialised (missing weights, missing library, etc.).
    """
    backend = config.detector_backend.lower().strip()
    detector_cls = _BACKEND_MAP.get(backend)
    if detector_cls is None:
        accepted = ", ".join(f'"{k}"' for k in _BACKEND_MAP)
        raise DetectorInitError(
            f"make_detector: unrecognised detector_backend '{config.detector_backend}'. "
            f"Accepted values are: {accepted}."
        )

    logger.info("Initialising face detector backend: '%s'", backend)
    # Constructor may raise DetectorInitError; let it propagate unchanged.
    return detector_cls()


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale image to grayscale uint8."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
