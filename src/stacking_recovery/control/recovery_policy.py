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
            or isinstance(confidence,(bool, np.bool_))
            or not isinstance(confidence, (int, float, np.floating))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(reason, FailureReason)
            or not isinstance(attempt, int)
            or not isinstance(max_retries, int)
            or attempt < 0
            or max_retries < 0):
        return RecoveryDecision.SAFE_STOP
    if ok and reason == FailureReason.OK:
        if confidence >= CONFIDENCE_MIN:
            return RecoveryDecision.PROCEED
        return RecoveryDecision.RETRY if attempt < max_retries else RecoveryDecision.SAFE_STOP
    if reason in (FailureReason.UNKNOWN, FailureReason.STUB):
        return RecoveryDecision.SAFE_STOP
    if reason == FailureReason.OK:
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
            FailureReason.TILTED_OBJECT,
    ):
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

    def after_grasp_close(self, ctx:ManipulationContext, wrist_image_bgr:np.ndarray,
                          ) -> tuple[RecoveryDecision, GraspCheckResult]:
        try:
            result = self._grasp_check(wrist_image_bgr)

            if (
                    not isinstance(result.seated, (bool, np.bool_))
                    or not isinstance(result.yaw_error_deg, (int, float, np.floating))
                    or isinstance(result.yaw_error_deg, (bool, np.bool_))
                    or not math.isfinite(result.yaw_error_deg)
                    or (result.seated and result.reason != FailureReason.OK)
                    or (not result.seated and result.reason == FailureReason.OK)
            ):
                raise ValueError("Invalid grasp check fields")
        except Exception:
            result = GraspCheckResult(
                seated=False,
                yaw_error_deg=0.0,
                confidence=0.0,
                reason=FailureReason.UNKNOWN,)

        decision = choose_action(
            ok=result.seated,
            confidence=result.confidence,
            reason=result.reason,
            attempt=ctx.grasp_attempt_index,
            max_retries=self.MAX_GRASP_RETRIES,)

        if decision is RecoveryDecision.PROCEED:
            outcome = "success"
        elif decision is RecoveryDecision.SAFE_STOP:
            outcome = "failed"
        else:
            outcome = "retry"

        self._log_grasp(ctx, result, outcome)

        if decision is RecoveryDecision.SAFE_STOP:
            self._log_stop(ctx, result.reason.value, attempt_index=ctx.grasp_attempt_index)
        elif decision is not RecoveryDecision.PROCEED:
            self._log_action(ctx, decision.value, result.reason.value, outcome, attempt_index=ctx.grasp_attempt_index)
        return decision, result

    def after_place(self, ctx: ManipulationContext, verify_image_bgr: np.ndarray,
                    ) -> tuple[RecoveryDecision, PlaceCheckResult]:
        try:
            result = self._place_check(verify_image_bgr, ctx.stack_level)

            if not isinstance(result, PlaceCheckResult):
                raise ValueError("Invalid placement check result")
            if (
                    not isinstance(result.placed, (bool, np.bool_))
                    or (result.placed and result.reason != FailureReason.OK)
                    or (not result.placed and result.reason == FailureReason.OK)
            ):
                raise ValueError("Invalid placement check fields")

        except Exception:
            result = PlaceCheckResult(
                placed=False,
                confidence=0.0,
                reason=FailureReason.UNKNOWN,)

        decision = choose_action(
            ok=result.placed,
            confidence=result.confidence,
            reason=result.reason,
            attempt=ctx.placement_attempt_index,
            max_retries=self.MAX_PLACEMENT_RETRIES,)

        if decision is RecoveryDecision.PROCEED:
            outcome = "success"
        elif decision is RecoveryDecision.SAFE_STOP:
            outcome = "failed"
        else:
            outcome = "retry"

        self._log_place(ctx, result, outcome)

        if decision is RecoveryDecision.SAFE_STOP:
            self._log_stop(ctx, result.reason.value, attempt_index=ctx.placement_attempt_index)
        elif decision is not RecoveryDecision.PROCEED:
            self._log_action(ctx, decision.value, result.reason.value, outcome, attempt_index=ctx.placement_attempt_index)
        return decision, result

    def _log_grasp(
            self,
            ctx: ManipulationContext,
            result: GraspCheckResult,
            outcome: str,
    ) -> None:
        self._log_action(
            ctx,
            "grasp_check",
            result.reason.value,
            outcome,
            attempt_index=ctx.grasp_attempt_index,
            seated=result.seated,
            check_confidence=result.confidence,
            yaw_error_deg=result.yaw_error_deg,
        )

    def _log_place(
            self,
            ctx: ManipulationContext,
            result: PlaceCheckResult,
            outcome: str,
    ) -> None:
        self._log_action(
            ctx,
            "place_check",
            result.reason.value,
            outcome,
            attempt_index=ctx.placement_attempt_index,
            placed=result.placed,
            check_confidence=result.confidence,
        )

    def _log_stop(
            self,
            ctx: ManipulationContext,
            reason: str,
            attempt_index: int,
    ) -> None:
        self._log_action(
            ctx,
            "safe_stop",
            reason,
            "failed",
            attempt_index=attempt_index,
        )

    def _log_action(
            self,
            ctx: ManipulationContext,
            action: str,
            reason: str,
            outcome: str,
            *,
            attempt_index: int,
            **kwargs: object,
    ) -> None:
        if not self._log_path:
            return

        event = new_event(
            cube_colour=ctx.cube_colour,
            stack_level=ctx.stack_level,
            attempt_index=attempt_index,
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