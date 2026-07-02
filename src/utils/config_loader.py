"""
src/utils/config_loader.py

Loads, validates, and merges ``config.yaml`` with CLI overrides.

Requirements: 14.1–14.6
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.utils.exceptions import ConfigError  # noqa: F401 – re-exported for callers
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_path: str
    dataset_type: str                           # "kaggle" | "vggface2"
    max_identities: int | None
    max_images_per_identity: int | None

    # ── Preprocessing ─────────────────────────────────────────────────────
    detector_backend: str                       # "haar"|"dnn"|"mtcnn"|"retinaface"
    no_face_fallback: str                       # "skip" | "use_full"
    image_size: tuple[int, int]                 # (width, height)

    # ── Deep-learning normalisation ───────────────────────────────────────
    norm_mean: list[float]                      # [R, G, B]
    norm_std: list[float]

    # ── Splitting ─────────────────────────────────────────────────────────
    split_ratios: tuple[float, float, float]    # (train, val, test)
    random_seed: int
    split_metadata_path: str
    force_resplit: bool

    # ── Classical pipeline ────────────────────────────────────────────────
    vocab_size: int
    kmeans_max_iter: int
    knn_k: int
    knn_metric: str
    bovw_artifact_path: str
    tfidf_artifact_path: str
    knn_artifact_path: str
    retrain: bool

    # ── Verification ──────────────────────────────────────────────────────
    verification_mode: bool
    verification_threshold: float
    ransac_enabled: bool

    # ── Deep learning ─────────────────────────────────────────────────────
    architecture: str                           # "resnet18"|"resnet50"|"mobilenet"|"efficientnet"
    pretrained: bool
    epochs: int
    optimizer: str
    learning_rate: float
    batch_size: int
    checkpoint_path: str
    eval_mode: str                              # "classification" | "embedding"
    embedding_classifier: str                  # "knn" | "cosine"
    cosine_threshold: float

    # ── Evaluation ────────────────────────────────────────────────────────
    roc_enabled: bool
    output_dir: str
    plots_dir: str
    results_dir: str
    artifacts_dir: str

    # ── Optional ArcFace ──────────────────────────────────────────────────
    arcface_enabled: bool
    arcface_margin: float
    arcface_scale: float


# ---------------------------------------------------------------------------
# Field type registry
# ---------------------------------------------------------------------------
# Maps each Config field name to the Python type we expect.
# Used both for required-field validation and CLI override coercion.

_FIELD_TYPES: dict[str, type] = {f.name: f.type for f in fields(Config)}  # type: ignore[misc]

# Exact runtime types to coerce into (mirrors the dataclass annotations).
_RUNTIME_TYPES: dict[str, Any] = {
    "dataset_path": str,
    "dataset_type": str,
    "max_identities": int,          # None handled separately
    "max_images_per_identity": int, # None handled separately
    "detector_backend": str,
    "no_face_fallback": str,
    "image_size": tuple,            # coerced via _coerce_sequence
    "norm_mean": list,
    "norm_std": list,
    "split_ratios": tuple,
    "random_seed": int,
    "split_metadata_path": str,
    "force_resplit": bool,
    "vocab_size": int,
    "kmeans_max_iter": int,
    "knn_k": int,
    "knn_metric": str,
    "bovw_artifact_path": str,
    "tfidf_artifact_path": str,
    "knn_artifact_path": str,
    "retrain": bool,
    "verification_mode": bool,
    "verification_threshold": float,
    "ransac_enabled": bool,
    "architecture": str,
    "pretrained": bool,
    "epochs": int,
    "optimizer": str,
    "learning_rate": float,
    "batch_size": int,
    "checkpoint_path": str,
    "eval_mode": str,
    "embedding_classifier": str,
    "cosine_threshold": float,
    "roc_enabled": bool,
    "output_dir": str,
    "plots_dir": str,
    "results_dir": str,
    "artifacts_dir": str,
    "arcface_enabled": bool,
    "arcface_margin": float,
    "arcface_scale": float,
}

# Fields whose value may legitimately be None (optional fields).
_NULLABLE_FIELDS: frozenset[str] = frozenset({"max_identities", "max_images_per_identity"})

# All required Config fields (non-nullable).
_REQUIRED_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(Config) if f.name not in _NULLABLE_FIELDS
)


# ---------------------------------------------------------------------------
# YAML key → Config field mapping
# ---------------------------------------------------------------------------
# The YAML file is nested; this flat mapping drives extraction.
# Key = dotted YAML path (section.key), Value = Config field name.

_YAML_PATH_TO_FIELD: dict[str, str] = {
    # dataset
    "dataset.path":                         "dataset_path",
    "dataset.type":                         "dataset_type",
    "dataset.max_identities":               "max_identities",
    "dataset.max_images_per_identity":      "max_images_per_identity",
    # preprocessing
    "preprocessing.detector_backend":       "detector_backend",
    "preprocessing.no_face_fallback":       "no_face_fallback",
    "preprocessing.image_size":             "image_size",
    "preprocessing.norm_mean":              "norm_mean",
    "preprocessing.norm_std":              "norm_std",
    # splitting
    "splitting.ratios":                     "split_ratios",
    "splitting.random_seed":                "random_seed",
    "splitting.metadata_path":              "split_metadata_path",
    "splitting.force_resplit":              "force_resplit",
    # classical
    "classical.vocab_size":                 "vocab_size",
    "classical.kmeans_max_iter":            "kmeans_max_iter",
    "classical.knn_k":                      "knn_k",
    "classical.knn_metric":                 "knn_metric",
    "classical.artifacts_dir":              "artifacts_dir",  # also used for bovw/tfidf/knn paths
    # verification
    "verification.enabled":                 "verification_mode",
    "verification.threshold":               "verification_threshold",
    "verification.ransac_enabled":          "ransac_enabled",
    # deep
    "deep.architecture":                    "architecture",
    "deep.pretrained":                      "pretrained",
    "deep.epochs":                          "epochs",
    "deep.optimizer":                       "optimizer",
    "deep.learning_rate":                   "learning_rate",
    "deep.batch_size":                      "batch_size",
    "deep.checkpoint_path":                 "checkpoint_path",
    "deep.eval_mode":                       "eval_mode",
    "deep.embedding_classifier":            "embedding_classifier",
    "deep.cosine_threshold":                "cosine_threshold",
    "deep.arcface_enabled":                 "arcface_enabled",
    "deep.arcface_margin":                  "arcface_margin",
    "deep.arcface_scale":                   "arcface_scale",
    # evaluation
    "evaluation.roc_enabled":               "roc_enabled",
    "evaluation.output_dir":                "output_dir",
    "evaluation.results_dir":               "results_dir",
    "evaluation.plots_dir":                 "plots_dir",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_nested(data: dict, dotted_path: str) -> Any:
    """Traverse a nested dict using a dotted key path.

    Returns the sentinel ``_MISSING`` if any key in the path is absent.
    """
    _MISSING = object()
    parts = dotted_path.split(".")
    current: Any = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
    return current.get(part)  # type: ignore[union-attr]


_MISSING = object()


def _get_nested_v2(data: dict, dotted_path: str) -> Any:
    """Traverse nested dict; returns ``_MISSING`` sentinel if path absent."""
    parts = dotted_path.split(".")
    current: Any = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _coerce_bool(value: str) -> bool:
    """Convert common string representations of booleans."""
    if isinstance(value, bool):
        return value
    low = value.strip().lower()
    if low in {"true", "1", "yes", "on"}:
        return True
    if low in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as bool")


def _coerce_sequence(value: Any, target: type) -> Any:
    """Coerce a value to list or tuple.

    Accepts YAML-parsed lists, Python literals in string form, or
    comma-separated strings.
    """
    if isinstance(value, (list, tuple)):
        return target(value)
    if isinstance(value, str):
        value = value.strip()
        # Try Python literal first (handles "[1, 2]" or "(1, 2)")
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return target(parsed)
        except (ValueError, SyntaxError):
            pass
        # Fall back to comma-split
        parts = [p.strip() for p in value.strip("[]()").split(",")]
        return target(parts)
    raise ValueError(f"Cannot coerce {value!r} to {target.__name__}")


def _coerce_value(field_name: str, value: Any) -> Any:
    """Coerce *value* to the type expected for *field_name*.

    Raises ``ConfigError`` with a descriptive message on failure.
    """
    if value is None and field_name in _NULLABLE_FIELDS:
        return None

    target = _RUNTIME_TYPES.get(field_name)
    if target is None:
        return value  # Unknown field — pass through unchanged.

    try:
        if target is bool:
            if isinstance(value, bool):
                return value
            return _coerce_bool(str(value))

        if target in (list, tuple):
            seq = _coerce_sequence(value, target)
            # Convert inner elements to appropriate primitive types.
            if field_name in ("image_size",):
                return tuple(int(x) for x in seq)
            if field_name in ("split_ratios",):
                return tuple(float(x) for x in seq)
            if field_name in ("norm_mean", "norm_std"):
                return [float(x) for x in seq]
            return seq

        if target is int:
            if isinstance(value, float) and value == int(value):
                return int(value)
            return int(value)

        if target is float:
            return float(value)

        if target is str:
            return str(value)

    except (ValueError, TypeError) as exc:
        raise ConfigError(
            f"CLI override for '{field_name}': cannot coerce value {value!r} "
            f"to expected type '{target.__name__}' — {exc}"
        ) from exc

    return value


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Loads ``config.yaml``, applies CLI overrides, and validates all fields.

    Example usage::

        loader = ConfigLoader()
        cfg = loader.load("config.yaml", overrides={"epochs": "5"})
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        config_path: str | Path = "config.yaml",
        overrides: dict[str, str] | None = None,
    ) -> Config:
        """Load and validate configuration.

        Args:
            config_path: Path to the YAML configuration file.
            overrides:   Flat dict of CLI override values, e.g.
                         ``{"epochs": "10", "learning_rate": "0.0001"}``.
                         Keys must match Config field names exactly.
                         Values are provided as strings and will be
                         coerced to the expected type.

        Returns:
            A fully populated and validated :class:`Config` instance.

        Raises:
            ConfigError: If the file is absent, a required key is missing,
                         or a CLI override cannot be coerced to the
                         expected type.
        """
        config_path = Path(config_path)
        overrides = overrides or {}

        # Requirement 14.6: raise before any processing if file is absent.
        if not config_path.exists():
            raise ConfigError(
                f"Configuration file not found: '{config_path.resolve()}'. "
                "Please create a 'config.yaml' at the project root."
            )

        raw = self._parse_yaml(config_path)
        flat = self._flatten(raw)
        flat = self._apply_overrides(flat, overrides)
        self._validate_required(flat)
        return self._build_config(flat)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_yaml(path: Path) -> dict:
        """Parse YAML file and return raw dict."""
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError(
                f"'{path}' did not parse to a YAML mapping — "
                f"got {type(data).__name__}."
            )
        return data

    @staticmethod
    def _flatten(raw: dict) -> dict[str, Any]:
        """Convert nested YAML dict to a flat Config-field dict.

        Also derives artifact paths from the classical.artifacts_dir
        when individual paths are not set explicitly.
        """
        flat: dict[str, Any] = {}

        for yaml_path, field_name in _YAML_PATH_TO_FIELD.items():
            val = _get_nested_v2(raw, yaml_path)
            if val is not _MISSING:
                flat[field_name] = val

        # Derive bovw/tfidf/knn artifact paths from artifacts_dir if not
        # provided as standalone keys in YAML.
        artifacts_dir = flat.get("artifacts_dir", "outputs/artifacts/classical")
        if "bovw_artifact_path" not in flat:
            flat["bovw_artifact_path"] = str(
                Path(artifacts_dir) / "kmeans_vocab.pkl"
            )
        if "tfidf_artifact_path" not in flat:
            flat["tfidf_artifact_path"] = str(
                Path(artifacts_dir) / "tfidf_transformer.pkl"
            )
        if "knn_artifact_path" not in flat:
            flat["knn_artifact_path"] = str(
                Path(artifacts_dir) / "knn_classifier.pkl"
            )

        # retrain defaults to False if not present in YAML
        if "retrain" not in flat:
            flat["retrain"] = False

        return flat

    @staticmethod
    def _apply_overrides(flat: dict[str, Any], overrides: dict[str, str]) -> dict[str, Any]:
        """Apply CLI overrides, coercing each value to the expected type.

        Requirement 14.4 / 14.5: CLI values take precedence; bad coercions
        raise ConfigError before any processing begins.
        """
        for key, raw_value in overrides.items():
            if key not in _RUNTIME_TYPES:
                logger.warning(
                    "CLI override key '%s' is not a recognised Config field — ignored.",
                    key,
                )
                continue
            try:
                flat[key] = _coerce_value(key, raw_value)
            except ConfigError:
                raise  # already has a descriptive message
        return flat

    @staticmethod
    def _validate_required(flat: dict[str, Any]) -> None:
        """Raise ConfigError for any missing required field.

        Requirement 14.3: descriptive error naming the missing parameter.
        """
        for field_name in _REQUIRED_FIELDS:
            if field_name not in flat:
                raise ConfigError(
                    f"Required configuration parameter '{field_name}' is missing "
                    f"from config.yaml. Please add it before running any pipeline stage."
                )

    @staticmethod
    def _build_config(flat: dict[str, Any]) -> Config:
        """Construct a :class:`Config` from the flat field dict, coercing
        types where YAML may have delivered slightly different representations
        (e.g., a list instead of a tuple for image_size).
        """
        kwargs: dict[str, Any] = {}
        for field_name in [f.name for f in fields(Config)]:
            raw_val = flat.get(field_name)
            if raw_val is None and field_name in _NULLABLE_FIELDS:
                kwargs[field_name] = None
            else:
                kwargs[field_name] = _coerce_value(field_name, raw_val)
        return Config(**kwargs)
