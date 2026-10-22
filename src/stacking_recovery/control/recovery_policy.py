"""Bounded regrasp after failed grasp or placement checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    RETRY = "retry"
    REGRASP = "regrasp"
    REACQUIRE = "reacquire"
    SAFE_STOP = "safe_stop"


@dataclass
class ManipulationContext:
    cube_colour: str
    stack_level: int
    table_pose_confidence: float
    grasp_attempt_index: int = 0
    placement_attempt_index: int = 0

def choose_action(
    ok: bool,
    confidence: float,
    reason: FailureReason,
    attempt: int,
    max_retries: int ) -> RecoveryDecision:
    if (
            not isinstance(ok, (bool, np.bool_))
            or not isinstance(confidence, (int, float, np.floating))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(reason, FailureReason)
            or not isinstance(attempt, int)
            or not isinstance(max_retries, int)
            or attempt < 0
            or max_retries < 0
    ):
        return RecoveryDecision.SAFE_STOP
    if ok and reason == FailureReason.OK and confidence >= CONFIDENCE_MIN:
        return RecoveryDecision.PROCEED
    if reason in (FailureReason.UNKNOWN, FailureReason.STUB):
        return RecoveryDecision.SAFE_STOP
    if attempt >= max_retries:
        return RecoveryDecision.SAFE_STOP
    if confidence < CONFIDENCE_MIN or reason == FailureReason.LOW_CONFIDENCE:
        return RecoveryDecision.RETRY
    if reason == FailureReason.OCCLUSION:
        return RecoveryDecision.REACQUIRE
    if reason in (
        FailureReason.EMPTY_GRASP,
        FailureReason.MISALIGNED_CUBE,
        FailureReason.TILTED_OBJECT):
        return RecoveryDecision.REGRASP
    if reason == FailureReason.PLACEMENT_FAILURE:
        return RecoveryDecision.RETRY
    return RecoveryDecision.SAFE_STOP


class RecoveryPolicy:
    MAX_GRASP_RETRIES = 2
    MAX_PLACEMENT_RETRIES = 1

    def __init__(
        self,
        log_path: Optional[Path] = None,
        grasp_check_fn: Callable[[np.ndarray], GraspCheckResult] = grasp_check_stub,
        place_check_fn: Callable[[np.ndarray, int], PlaceCheckResult] = place_check_stub,
    ) -> None:
        self._log_path = log_path
        self._grasp_check = grasp_check_fn
        self._place_check = place_check_fn

    def after_grasp_close(self, ctx, wrist_image_bgr):
        result = self._grasp_check(wrist_image_bgr)
        decision = choose_action(
            ok=result.seated,
            confidence=result.confidence,
            reason=result.reason,
            attempt=ctx.grasp_attempt_index,
            max_retries=self.MAX_GRASP_RETRIES,
        )
        return decision, result

    def after_place(
        self,
        ctx: ManipulationContext,
        verify_image_bgr: np.ndarray,
    ) -> tuple[RecoveryDecision, PlaceCheckResult]:
        result = self._place_check(verify_image_bgr, ctx.stack_level)
        self._log_place(ctx, result)

        decision = choose_action(
            ok=result.placed,
            confidence=result.confidence,
            reason=result.reason,
            attempt=ctx.placement_attempt_index,
            max_retries=self.MAX_PLACEMENT_RETRIES,
        )
        if decision is RecoveryDecision.SAFE_STOP:
            self._log_stop(ctx, result.reason.value)
        elif decision is not RecoveryDecision.PROCEED:
            self._log_action(ctx, decision.value, result.reason.value, "retry")
        return decision, result


    def _log_grasp(
        self, ctx: ManipulationContext, result: GraspCheckResult, outcome: str
    ) -> None:
        self._log_action(
            ctx,
            "grasp_check",
            result.reason.value,
            outcome,  # type: ignore[arg-type]
            seated=result.seated,
            check_confidence=result.confidence,
            yaw_error_deg=result.yaw_error_deg,
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
        )
        append_event(self._log_path, event)
