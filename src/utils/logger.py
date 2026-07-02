"""
src/utils/logger.py

Module-level logging configuration for the facial recognition comparison system.
Provides a consistent log format (timestamp, level, module name) and a
get_logger() factory used by all other modules.

Requirements: 17.1–17.4
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Root logger configuration
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure the root logger once at module import time.
# Only add a StreamHandler if none are already present (avoids duplicate
# handlers when the module is reloaded in tests or Jupyter notebooks).
_root = logging.getLogger("facial_recognition")
if not _root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    _root.addHandler(_handler)
    _root.setLevel(logging.DEBUG)
    # Do not propagate to the Python root logger to avoid double output
    _root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``facial_recognition`` namespace.

    All child loggers inherit the handlers and formatter configured on the
    ``facial_recognition`` root logger, so every message will carry the
    consistent format::

        2024-01-01 12:00:00  WARNING   facial_recognition.preprocessing  <msg>

    Args:
        name: Typically ``__name__`` of the calling module, e.g.
              ``"src.preprocessing.dataset_loader"``.  The final logger name
              will be ``facial_recognition.<name>``.

    Returns:
        A :class:`logging.Logger` instance ready for use.

    Example::

        from src.utils.logger import get_logger

        logger = get_logger(__name__)
        logger.info("DatasetLoader initialised")
        logger.warning("No face detected — image skipped: %s", path)
    """
    return logging.getLogger(f"facial_recognition.{name}")
