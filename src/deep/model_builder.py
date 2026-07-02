"""
src/deep/model_builder.py

Builds the deep learning backbone with a custom classification head.

Requirements: 9.1–9.4
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from src.utils.exceptions import ModelConfigError
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.config_loader import Config

logger = get_logger(__name__)

_VALID_ARCHITECTURES = frozenset({"resnet18", "resnet50", "mobilenet", "efficientnet"})


class ModelBuilder:
    """Constructs a CNN backbone with a custom classification head."""

    def build(self, config: "Config", num_classes: int) -> nn.Module:
        """Build and return the model.

        Args:
            config: Validated Config instance.
            num_classes: Number of identity classes in the training set.

        Returns:
            A PyTorch nn.Module ready for training.

        Raises:
            ModelConfigError: If architecture is unrecognized.
        """
        arch = config.architecture.lower().strip()

        if arch not in _VALID_ARCHITECTURES:
            raise ModelConfigError(
                f"Unrecognised architecture '{config.architecture}'. "
                f"Accepted values: {sorted(_VALID_ARCHITECTURES)}."
            )

        pretrained = config.pretrained
        logger.info("Building model: arch=%s, pretrained=%s, num_classes=%d", arch, pretrained, num_classes)

        try:
            import torchvision.models as models
        except ImportError as exc:
            raise ModelConfigError("torchvision is required for deep learning pipeline.") from exc

        if arch == "resnet18":
            weights = "IMAGENET1K_V1" if pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif arch == "resnet50":
            weights = "IMAGENET1K_V1" if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif arch == "mobilenet":
            weights = "IMAGENET1K_V1" if pretrained else None
            model = models.mobilenet_v2(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

        elif arch == "efficientnet":
            weights = "IMAGENET1K_V1" if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

        logger.info("Model built successfully.")
        return model
