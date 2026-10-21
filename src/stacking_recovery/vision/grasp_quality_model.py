"""MobileNetV3 grasp-quality classifier — train offline, infer at runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from stacking_recovery.types import FailureReason, GraspCheckResult
from stacking_recovery.vision.grasp_labels import (
    CLASS_TO_INDEX,
    GRASP_CLASSES,
    class_to_failure_reason,
    is_seated_class,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import torch
    import torch.nn as nn
    from torchvision import models
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    models = None  # type: ignore


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_grasp_config(path: Optional[Path] = None) -> dict[str, Any]:
    cfg_path = path or repo_root() / "configs" / "grasp_model.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def crop_roi(image_bgr: np.ndarray, roi: dict[str, float]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    y0 = int(h * roi["y0"])
    y1 = int(h * roi["y1"])
    x0 = int(w * roi["x0"])
    x1 = int(w * roi["x1"])
    return image_bgr[y0:y1, x0:x1]


def crop_for_observation(
    image_bgr: np.ndarray,
    roi: dict[str, float],
    observation_source: Optional[str] = None,
) -> np.ndarray:
    """Keep prepared table-camera pairs intact; crop legacy wrist frames."""
    if observation_source == "table_before_after":
        return image_bgr
    return crop_roi(image_bgr, roi)


def preprocess(image_bgr: np.ndarray, size: int) -> "torch.Tensor":
    if torch is None or cv2 is None:
        raise RuntimeError("torch and opencv required for ML grasp check")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def build_mobilenet_v3(num_classes: int, *, pretrained: bool = True) -> "nn.Module":
    if models is None:
        raise RuntimeError("torchvision required")
    weights = (
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    net = models.mobilenet_v3_small(weights=weights)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    return net


@dataclass(frozen=True)
class GraspQualityPrediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class GraspQualityModel:
    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or load_grasp_config()
        self._model: Optional[nn.Module] = None
        self._device = torch.device("cpu") if torch else None

    @property
    def model_path(self) -> Path:
        p = Path(self.config["model_path"])
        if not p.is_absolute():
            p = repo_root() / p
        return p

    def load(self) -> None:
        if torch is None:
            raise RuntimeError("Install torch: pip install -e '.[ml]'")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Grasp model not found: {self.model_path}. "
                "Run scripts/train_grasp_classifier.py first."
            )
        num_classes = len(self.config.get("classes", GRASP_CLASSES))
        # The checkpoint already contains every weight. Avoid an internet download
        # when the model is loaded in the hackathon lab or on a fresh Brev instance.
        self._model = build_mobilenet_v3(num_classes, pretrained=False)
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        observation_source: Optional[str] = None,
    ) -> GraspQualityPrediction:
        if self._model is None:
            self.load()
        assert self._model is not None and self._device is not None
        roi = crop_for_observation(
            image_bgr, self.config["roi"], observation_source
        )
        size = int(self.config.get("input_size", 224))
        x = preprocess(roi, size).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        name = GRASP_CLASSES[idx] if idx < len(GRASP_CLASSES) else "UNCERTAIN"
        prob_map = {
            GRASP_CLASSES[i]: float(probs[i]) for i in range(min(len(probs), len(GRASP_CLASSES)))
        }
        return GraspQualityPrediction(
            class_name=name,
            confidence=float(probs[idx]),
            probabilities=prob_map,
        )


_default_model: Optional[GraspQualityModel] = None


def get_grasp_quality_model() -> GraspQualityModel:
    global _default_model
    if _default_model is None:
        _default_model = GraspQualityModel()
    return _default_model


def prediction_to_grasp_result(pred: GraspQualityPrediction, config: dict[str, Any]) -> GraspCheckResult:
    tau_good = float(config.get("thresholds", {}).get("tau_good", 0.75))
    tau_act = float(config.get("thresholds", {}).get("tau_act", 0.55))
    name = pred.class_name
    conf = pred.confidence
    reason = class_to_failure_reason(name)

    meta = {"predicted_class": name, "verification": "ai"}
    if name == "GOOD" and conf >= tau_good:
        return GraspCheckResult(
            seated=True, yaw_error_deg=0.0, confidence=conf, reason=reason, **meta
        )
    if conf < tau_act or name == "UNCERTAIN":
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=float("nan"),
            confidence=conf,
            reason=FailureReason.LOW_CONFIDENCE,
            **meta,
        )
    return GraspCheckResult(
        seated=False,
        yaw_error_deg=float("nan"),
        confidence=conf,
        reason=reason,
        **meta,
    )


def grasp_check_ml(image_bgr: np.ndarray) -> GraspCheckResult:
    """AI grasp verification entry point."""
    cfg = load_grasp_config()
    pred = get_grasp_quality_model().predict(image_bgr)
    return prediction_to_grasp_result(pred, cfg)


def save_model_checkpoint(model: "nn.Module", path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
