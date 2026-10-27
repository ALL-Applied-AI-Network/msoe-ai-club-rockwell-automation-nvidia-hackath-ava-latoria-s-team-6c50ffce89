"""Bounded regrasp after failed grasp or placement checks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from stacking_recovery.events import append_event, new_event
from stacking_recovery.types import (
    CONFIDENCE_MIN,
    FailureReason,
    GraspCheckResult,
    PlaceCheckResult,
)
from stacking_recovery.vision.grasp_check import grasp_check_stub
from stacking_recovery.vision.place_check import place_check_stub


class RecoveryDecision(str, Enum):
    PROCEED = "proceed"
    SECOND_LOOK = "second_look"
    REACQUIRE = "reacquire"
    REGRASP = "regrasp"
    SAFE_STOP = "safe_stop"


@dataclass
class ManipulationContext:
    cube_colour: str
    stack_level: int
    table_pose_confidence: float
    grasp_attempt_index: int = 0
    placement_attempt_index: int = 0
    uncertain_look_index: int = 0
    pick_started_at: float = field(default_factory=time.perf_counter)


class RecoveryPolicy:
    MAX_GRASP_RETRIES = 2
    MAX_PLACEMENT_RETRIES = 1
    MAX_UNCERTAIN_LOOKS = 1

    def __init__(
        self,
        log_path: Optional[Path] = None,
        grasp_check_fn: Callable[[np.ndarray], GraspCheckResult] = grasp_check_stub,
        place_check_fn: Callable[[np.ndarray, int], PlaceCheckResult] = place_check_stub,
    ) -> None:
        self._log_path = log_path
        self._grasp_check = grasp_check_fn
        self._place_check = place_check_fn

    def after_grasp_close(
        self,
        ctx: ManipulationContext,
        wrist_image_bgr: np.ndarray,
    ) -> tuple[RecoveryDecision, GraspCheckResult]:
        result = self._grasp_check(wrist_image_bgr)
        self._log_grasp(ctx, result, outcome="success" if result.seated else "retry")
        if result.seated and result.confidence >= CONFIDENCE_MIN:
            return RecoveryDecision.PROCEED, result
        predicted_class = result.predicted_class or ""
        if (
            result.reason == FailureReason.LOW_CONFIDENCE
            or predicted_class == "UNCERTAIN"
        ) and ctx.uncertain_look_index < self.MAX_UNCERTAIN_LOOKS:
            self._log_action(ctx, "second_look", result.reason.value, "retry")
            return RecoveryDecision.SECOND_LOOK, result
        if ctx.grasp_attempt_index >= self.MAX_GRASP_RETRIES:
            self._log_stop(ctx, FailureReason.MAX_RETRIES.value)
            return RecoveryDecision.SAFE_STOP, result
        if result.reason == FailureReason.EMPTY_GRASP or predicted_class == "EMPTY_GRASP":
            self._log_action(ctx, "reacquire", result.reason.value, "retry")
            return RecoveryDecision.REACQUIRE, result
        self._log_action(ctx, "regrasp", result.reason.value, "retry")
        return RecoveryDecision.REGRASP, result

    def after_place(
        self,
        ctx: ManipulationContext,
        verify_image_bgr: np.ndarray,
    ) -> tuple[RecoveryDecision, PlaceCheckResult]:
        result = self._place_check(verify_image_bgr, ctx.stack_level)
        self._log_place(ctx, result)
        # ``place_check`` owns the calibrated confidence threshold and only
        # marks a result placed after its geometry, support and confidence
        # gates pass.  Applying the unrelated global grasp threshold here can
        # turn a valid ``placed=True`` result into a contradictory regrasp.
        if result.placed:
            self._log_pick_duration(ctx)
            return RecoveryDecision.PROCEED, result
        if ctx.placement_attempt_index >= self.MAX_PLACEMENT_RETRIES:
            self._log_stop(ctx, FailureReason.PLACEMENT_FAILURE.value)
            return RecoveryDecision.SAFE_STOP, result
        self._log_action(ctx, "regrasp", result.reason.value, "retry")
        return RecoveryDecision.REGRASP, result

    def _log_pick_duration(self, ctx: ManipulationContext) -> None:
        if not self._log_path:
            return
        duration = time.perf_counter() - ctx.pick_started_at
        self._log_action(
            ctx,
            "place",
            FailureReason.OK.value,
            "success",
            duration_s=duration,
        )

    def _log_grasp(
        self, ctx: ManipulationContext, result: GraspCheckResult, outcome: str
    ) -> None:
        extra: dict[str, object] = {}
        if result.predicted_class is not None:
            extra["predicted_class"] = result.predicted_class
        if result.verification != "rules":
            extra["verification"] = result.verification
        self._log_action(
            ctx,
            "grasp_check",
            result.reason.value,
            outcome,  # type: ignore[arg-type]
            seated=result.seated,
            check_confidence=result.confidence,
            yaw_error_deg=result.yaw_error_deg,
            extra=extra or None,
        )

    def _log_place(self, ctx: ManipulationContext, result: PlaceCheckResult) -> None:
        self._log_action(
            ctx,
            "place_check",
            result.reason.value,
            "success" if result.placed else "retry",
            placed=result.placed,
            check_confidence=result.confidence,
        )

    def _log_stop(self, ctx: ManipulationContext, reason: str) -> None:
        self._log_action(ctx, "safe_stop", reason, "failed")

    def _log_action(
        self,
        ctx: ManipulationContext,
        action: str,
        reason: str,
        outcome: str,
        **kwargs: object,
    ) -> None:
        if not self._log_path:
            return
        event = new_event(
            cube_colour=ctx.cube_colour,
            stack_level=ctx.stack_level,
            attempt_index=ctx.grasp_attempt_index,
            table_pose_confidence=ctx.table_pose_confidence,
            action=action,  # type: ignore[arg-type]
            reason=reason,
            outcome=outcome,  # type: ignore[arg-type]
            seated=kwargs.get("seated"),  # type: ignore[arg-type]
            placed=kwargs.get("placed"),  # type: ignore[arg-type]
            check_confidence=kwargs.get("check_confidence"),  # type: ignore[arg-type]
            yaw_error_deg=kwargs.get("yaw_error_deg"),  # type: ignore[arg-type]
            duration_s=kwargs.get("duration_s"),  # type: ignore[arg-type]
            placement_attempt_index=ctx.placement_attempt_index,
            extra=dict(kwargs["extra"]) if kwargs.get("extra") else None,  # type: ignore[arg-type]
        )
        append_event(self._log_path, event)
