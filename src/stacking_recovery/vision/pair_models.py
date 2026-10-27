"""Pair-aware image models for table-camera grasp verification.

The table dataset stores a before crop and an after crop side-by-side.  A
Siamese model is a natural fit: the same encoder processes both images and a
small head classifies their feature difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    pair_mode: str
    pretrained: bool = True


def _weights(architecture: str, pretrained: bool):
    if not pretrained:
        return None
    return {
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    }[architecture]


def build_encoder(architecture: str, *, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return an ImageNet encoder whose output is a flat feature vector."""
    weights = _weights(architecture, pretrained)
    if architecture == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights=weights)
        feature_dim = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return net, feature_dim
    if architecture == "resnet18":
        net = models.resnet18(weights=weights)
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feature_dim
    if architecture == "efficientnet_b0":
        net = models.efficientnet_b0(weights=weights)
        feature_dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        return net, feature_dim
    raise ValueError(f"unsupported architecture: {architecture}")


def build_single_image_model(
    architecture: str, num_classes: int, *, pretrained: bool = True
) -> nn.Module:
    """Build a conventional classifier for a three-channel delta image."""
    weights = _weights(architecture, pretrained)
    if architecture == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        return net
    if architecture == "resnet18":
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if architecture == "efficientnet_b0":
        net = models.efficientnet_b0(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        return net
    raise ValueError(f"unsupported architecture: {architecture}")


class SiamesePairClassifier(nn.Module):
    """Shared RGB encoder with before, after, and absolute-difference features."""

    def __init__(
        self,
        architecture: str,
        num_classes: int,
        *,
        pretrained: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder, feature_dim = build_encoder(
            architecture, pretrained=pretrained
        )
        hidden = max(128, min(512, feature_dim))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 4 or pair.shape[1] != 6:
            raise ValueError("SiamesePairClassifier expects [N, 6, H, W]")
        before, after = pair[:, :3], pair[:, 3:]
        before_features = self.encoder(before)
        after_features = self.encoder(after)
        combined = torch.cat(
            [before_features, after_features, (after_features - before_features).abs()],
            dim=1,
        )
        return self.classifier(combined)


def build_pair_model(spec: ModelSpec, num_classes: int) -> nn.Module:
    if spec.pair_mode == "siamese":
        return SiamesePairClassifier(
            spec.architecture, num_classes, pretrained=spec.pretrained
        )
    if spec.pair_mode == "delta":
        return build_single_image_model(
            spec.architecture, num_classes, pretrained=spec.pretrained
        )
    raise ValueError(f"unsupported pair mode: {spec.pair_mode}")


def classifier_parameters(model: nn.Module, pair_mode: str):
    """Parameters to train during the short head-only warmup."""
    if pair_mode == "siamese":
        return model.classifier.parameters()
    if hasattr(model, "classifier"):
        return model.classifier.parameters()
    if hasattr(model, "fc"):
        return model.fc.parameters()
    raise ValueError("model has no recognized classifier head")
