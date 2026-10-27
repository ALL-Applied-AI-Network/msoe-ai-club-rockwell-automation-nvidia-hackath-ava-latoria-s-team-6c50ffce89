"""Runtime + Isaac grasp sample recording for continual improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from stacking_recovery.data.grasp_dataset_store import GraspDatasetStore, resolve_data_root
from stacking_recovery.data.isaac_grasp_labels import (
    label_from_physics_event,
    should_add_feedback_sample,
)
from stacking_recovery.types import GraspCheckResult

RecordMode = Literal["off", "feedback", "collect_all"]


@dataclass
class PendingGraspCapture:
    image_bgr: np.ndarray
    grasp_result: GraspCheckResult
    meta: dict[str, Any] = field(default_factory=dict)


class GraspFeedbackRecorder:
    """
    At dwell: stash the selected observation + verifier output.
    At lift: optionally make a table-camera before/after pair, apply the physics
    weak label, and save per mode (Isaac collect vs runtime feedback).
    """

    def __init__(
        self,
        store: GraspDatasetStore,
        *,
        mode: RecordMode = "feedback",
        record_bad_grasps: bool = True,
        record_mismatch: bool = True,
    ) -> None:
        self.store = store
        self.mode = mode
        self.record_bad_grasps = record_bad_grasps
        self.record_mismatch = record_mismatch
        self._pending: Optional[PendingGraspCapture] = None
        self.saved_count = 0

    @classmethod
    def from_config(cls, repo_root: Path, cfg: dict[str, Any]) -> Optional["GraspFeedbackRecorder"]:
        fb = cfg.get("feedback") or {}
        if not fb.get("enabled"):
            return None
        configured_root = fb.get("data_root")
        root = resolve_data_root(
            repo_root,
            Path(configured_root) if configured_root else None,
        )
        mode = fb.get("mode", "feedback")
        if mode not in ("feedback", "collect_all"):
            mode = "feedback"
        return cls(
            GraspDatasetStore(root),
            mode=mode,  # type: ignore[arg-type]
            record_bad_grasps=bool(fb.get("record_bad_grasps", True)),
            record_mismatch=bool(fb.get("record_mismatch", True)),
        )

    def on_dwell(
        self,
        image_bgr: np.ndarray,
        grasp_result: GraspCheckResult,
        **meta: Any,
    ) -> None:
        if self.mode == "off":
            return
        self._pending = PendingGraspCapture(
            image_bgr=image_bgr.copy(),
            grasp_result=grasp_result,
            meta=dict(meta),
        )

    def on_lift(
        self,
        event: dict[str, Any],
        *,
        after_image_bgr: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Call after physics pickup check; returns saved relative path if any."""
        if self.mode == "off" or self._pending is None:
            return None
        pending = self._pending
        self._pending = None

        if pending.meta.get("observation_source") == "table_before_after":
            if after_image_bgr is None:
                return None
            from stacking_recovery.data.table_grasp_observation import (
                make_before_after_observation,
            )

            center = pending.meta.get("crop_center_uv")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                return None
            pair, quality = make_before_after_observation(
                pending.image_bgr,
                after_image_bgr,
                (float(center[0]), float(center[1])),
                crop_size=int(pending.meta.get("crop_size_px", 256)),
            )
            pending.image_bgr = pair
            pending.meta["observation_quality"] = quality.to_dict()
            if not quality.valid:
                return None

        physics_label = label_from_physics_event(
            pickup_verified=bool(event.get("pickup_verified")),
            perceived_yaw_deg=event.get("perceived_yaw_deg"),
            yaw_error_deg=pending.meta.get("injected_yaw_error_deg"),
            jaw_gap_mm=event.get("jaw_gap_mm"),
            closed_loop_grasp=event.get("closed_loop_grasp"),
        )

        if self.mode == "collect_all":
            return self._save(pending, physics_label, event, source="isaac")

        ok, label = should_add_feedback_sample(
            physics_label=physics_label,
            grasp_result=pending.grasp_result,
            pickup_verified=bool(event.get("pickup_verified")),
            record_bad_grasps=self.record_bad_grasps,
            record_mismatch=self.record_mismatch,
        )
        if not ok:
            return None
        return self._save(pending, label, event, source="feedback")

    def on_grasp_failure_no_lift(
        self,
        image_bgr: np.ndarray,
        grasp_result: GraspCheckResult,
        **meta: Any,
    ) -> Optional[str]:
        """Safe-stop / regrasp abort before lift — still valuable for training."""
        if self.mode != "feedback" or not self.record_bad_grasps:
            return None
        if grasp_result.seated:
            return None
        from stacking_recovery.data.isaac_grasp_labels import grasp_result_to_class

        label = grasp_result_to_class(grasp_result)
        pending = PendingGraspCapture(image_bgr, grasp_result, meta=dict(meta))
        return self._save(pending, label, meta, source="feedback")

    def _save(
        self,
        pending: PendingGraspCapture,
        label: str,
        event: dict[str, Any],
        *,
        source: str,
    ) -> str:
        meta = {
            **pending.meta,
            "physics_label": label,
            "pickup_verified": event.get("pickup_verified"),
            "predicted_class": pending.grasp_result.predicted_class,
            "verifier_seated": pending.grasp_result.seated,
            "verification": pending.grasp_result.verification,
        }
        if event.get("detection_id") is not None:
            meta["detection_id"] = event["detection_id"]
        if event.get("stack_level") is not None:
            meta["stack_level"] = event["stack_level"]
        split = str(pending.meta.get("split", "train"))
        rec = self.store.save_sample(
            pending.image_bgr,
            label,
            source=source,
            meta=meta,
            split=split,
        )
        self.saved_count += 1
        return rec.path
