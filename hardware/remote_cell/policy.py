"""Conservative policy for the official remote FANUC cell client.

Copy this file over the official client's ``policy.py``.  It deliberately uses
the official calibration module and TCP action API instead of connecting to a
robot directly.  The first physical rollout should stay supervised and should
not enable the learned placement correction until its camera calibration has
been checked on real images.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


TCP_BOX = np.array([[150.0, 650.0], [-450.0, 450.0], [-334.0, 300.0]])
SAFE_Z_MM = -250.0
PICK_Z_MM = -329.0
PLACE_Z_MM = -319.0
MAX_REACH_MM = 550.0
OPERATOR_ANCHOR_TOLERANCE_MM = 75.0
TELEMETRY_ROOT = Path(
    os.environ.get("REMOTE_ARM_TELEMETRY_DIR", "/workspace/team-data/remote-arm-telemetry")
)
FIXED_TARGET_ENV = {
    "red": "REMOTE_ARM_RED_XY_MM",
    "blue": "REMOTE_ARM_BLUE_XY_MM",
}


def component_centers(
    mask: np.ndarray,
    *,
    min_area: int = 10,
    max_area_fraction: float = 0.02,
) -> list[dict]:
    """Return compact, non-border color components with simple shape scores."""

    binary = np.asarray(mask, dtype=np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return []
    height, width = binary.shape[:2]
    candidates: list[dict] = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        box_w = int(stats[index, cv2.CC_STAT_WIDTH])
        box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or area > binary.size * max_area_fraction:
            continue
        # The cube can be small, but a valid component should not be a stripe,
        # sparse reflection, or a colored region clipped by the image edge.
        if x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1:
            continue
        aspect = box_w / max(1.0, float(box_h))
        fill = area / max(1.0, float(box_w * box_h))
        if not 0.35 <= aspect <= 2.85 or fill < 0.25:
            continue
        u, v = centers[index]
        shape_score = fill / (1.0 + abs(math.log(aspect)))
        candidates.append(
            {
                "center": (float(u), float(v)),
                "area": area,
                "bbox": [x, y, box_w, box_h],
                "shape_score": float(shape_score),
            }
        )
    return sorted(candidates, key=lambda item: (item["area"], item["shape_score"]), reverse=True)


def largest_component_center(mask: np.ndarray, *, min_area: int = 10) -> tuple[float, float] | None:
    """Return the largest compact color component's (u, v), or ``None``."""

    candidates = component_centers(mask, min_area=min_area, max_area_fraction=0.08)
    return candidates[0]["center"] if candidates else None


def color_components(
    image: np.ndarray,
    color: str,
    *,
    max_area_fraction: float = 0.02,
) -> list[dict]:
    """Find compact RGB color components without assuming an image location."""

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return []
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    if color == "red":
        mask = (r > 85) & (r > g * 1.35) & (r > b * 1.35)
    elif color == "blue":
        mask = (b > 75) & (b > g * 1.25) & (b > r * 1.30)
    else:
        raise ValueError(f"unsupported color: {color}")
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return component_centers(mask, max_area_fraction=max_area_fraction)


def color_center(image: np.ndarray, color: str) -> tuple[float, float] | None:
    """Return the strongest compact RGB component for wrist-camera checks."""

    # A grasped cube can fill much more of the wrist frame than a cube in the
    # overhead table frame, so this verification path intentionally allows a
    # larger compact component.
    candidates = color_components(image, color, max_area_fraction=0.25)
    return candidates[0]["center"] if candidates else None


def parse_xy(value: str | None) -> tuple[float, float] | None:
    """Parse an explicit ``x,y`` millimeter target from the environment."""

    if not value:
        return None
    try:
        fields = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(f"invalid XY target {value!r}; expected x,y in millimeters") from error
    if len(fields) != 2 or not np.all(np.isfinite(fields)):
        raise ValueError(f"invalid XY target {value!r}; expected two finite values")
    return float(fields[0]), float(fields[1])


def tcp_is_safe(target: Iterable[float]) -> bool:
    values = np.asarray(list(target), dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        return False
    if np.any(values[:3] < TCP_BOX[:, 0]) or np.any(values[:3] > TCP_BOX[:, 1]):
        return False
    if math.hypot(values[0], values[1]) > MAX_REACH_MM:
        return False
    return bool(np.max(np.abs(values[3:6])) <= 360.0 and 0.0 <= values[6] <= 1.0)


def interpolate_tcp(start: Iterable[float] | None, target: Iterable[float], *, step_mm: float = 10.0) -> list[list[float]]:
    """Create a bounded, slow Cartesian chunk with a constant tool attitude."""

    goal = np.asarray(list(target), dtype=float)
    if not tcp_is_safe(goal):
        raise ValueError(f"unsafe TCP target: {goal.tolist()}")
    if start is None:
        return [goal.tolist()]
    initial = np.asarray(list(start), dtype=float)
    if initial.shape != (6,) or not np.all(np.isfinite(initial)):
        return [goal.tolist()]
    distance = float(np.linalg.norm(goal[:3] - initial[:3]))
    steps = max(1, min(100, int(math.ceil(distance / step_mm))))
    rows: list[list[float]] = []
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        xyz = initial[:3] + alpha * (goal[:3] - initial[:3])
        row = [*xyz.tolist(), *goal[3:]]
        if not tcp_is_safe(row):
            raise ValueError(f"unsafe interpolated TCP target: {row}")
        rows.append(row)
    return rows


class Policy:
    """Pick the red 10 mm cube and place it on the blue 10 mm cube."""

    CAMERAS = ["table", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH: list[str] = []
    DT = 0.2

    def __init__(self, checkpoint: str | None = None):
        _ = checkpoint
        self._calibration_error: str | None = None
        try:
            from calibration.geometry import load, pixel_to_base

            self._cams = load()
            self._pixel_to_base = pixel_to_base
        except Exception as error:  # fail closed if copied outside the official client
            self._cams = None
            self._pixel_to_base = None
            self._calibration_error = str(error)
        self._fixed_targets = {
            color: parse_xy(os.environ.get(variable))
            for color, variable in FIXED_TARGET_ENV.items()
        }
        configured = [color for color, xy in self._fixed_targets.items() if xy is not None]
        if configured and len(configured) != len(self._fixed_targets):
            raise ValueError(
                "operator fallback requires both REMOTE_ARM_RED_XY_MM and "
                "REMOTE_ARM_BLUE_XY_MM"
            )
        for color, xy in self._fixed_targets.items():
            if xy is not None and not tcp_is_safe(self._target(xy, SAFE_Z_MM, 0.0)):
                raise ValueError(f"unsafe configured {color} target: {xy}")
        self._operator_fallback_active = bool(configured)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._session_id = f"remote-{stamp}-{os.getpid()}-{time.time_ns() % 1_000_000_000:09d}"
        self._telemetry_dir = TELEMETRY_ROOT / self._session_id
        self._events_path = self._telemetry_dir / "events.jsonl"
        self._latest_path = self._telemetry_dir / "latest.json"
        self._summary_path = self._telemetry_dir / "summary.json"
        self._telemetry_error: str | None = None
        self._rollout_index = -1
        self._total_steps = 0
        try:
            self._telemetry_dir.mkdir(parents=True, exist_ok=False)
            (TELEMETRY_ROOT / "current_session.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "session_id": self._session_id,
                        "path": str(self._telemetry_dir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as error:
            self._telemetry_error = str(error)
        self.reset()

    def reset(self) -> None:
        self._rollout_index += 1
        self._step = 0
        self._last_detections: dict[str, dict] = {}
        self._last_wrist_red_visible = False
        self.phase = "locate"
        self.red_xy_mm: tuple[float, float] | None = None
        self.blue_xy_mm: tuple[float, float] | None = None
        self.result = "pending"
        self.reason: str | None = None

    @staticmethod
    def _target(xy: tuple[float, float], z: float, gripper: float) -> list[float]:
        return [xy[0], xy[1], z, -180.0, 0.0, 0.0, gripper]

    @staticmethod
    def _current_pose(obs: dict) -> list[float] | None:
        pose = obs.get("tcp_pose")
        if pose is None:
            return None
        values = np.asarray(pose, dtype=float).reshape(-1)
        return values[:6].tolist() if values.size >= 6 and np.all(np.isfinite(values[:6])) else None

    def _hold(self, obs: dict, *, closed: bool = False) -> list[float]:
        current = self._current_pose(obs)
        if current is None or not tcp_is_safe([*current, float(closed)]):
            return [450.0, 0.0, SAFE_Z_MM, -180.0, 0.0, 0.0, float(closed)]
        return [*current, float(closed)]

    def _table_xy(
        self,
        image: np.ndarray,
        color: str,
        *,
        allow_operator_fallback: bool = True,
    ) -> tuple[float, float] | None:
        if self._cams is None or self._pixel_to_base is None:
            self._last_detections[color] = {"visible": False, "reason": "calibration_unavailable"}
            return None
        components = color_components(image, color)
        if not components:
            self._last_detections[color] = {"visible": False}
            return None
        native_w, native_h = self._cams["table"]["size"]
        height, width = image.shape[:2]
        fallback = self._fixed_targets.get(color)
        evaluated: list[dict] = []
        for component in components:
            center = component["center"]
            uv_native = (center[0] * native_w / width, center[1] * native_h / height)
            xyz_m = self._pixel_to_base(self._cams["table"], uv_native)
            xy = float(xyz_m[0] * 1000.0), float(xyz_m[1] * 1000.0)
            safe = tcp_is_safe(self._target(xy, SAFE_Z_MM, 0.0))
            candidate = {
                "pixel": [float(center[0]), float(center[1])],
                "native_pixel": [float(uv_native[0]), float(uv_native[1])],
                "base_xy_mm": [float(xy[0]), float(xy[1])],
                "safe": bool(safe),
                "area": int(component["area"]),
                "bbox": component["bbox"],
                "shape_score": float(component["shape_score"]),
            }
            candidate["anchor_error_mm"] = (
                float(math.dist(xy, fallback)) if fallback is not None else None
            )
            evaluated.append(candidate)

        safe_candidates = [candidate for candidate in evaluated if candidate["safe"]]
        if fallback is not None:
            # A calibrated component that is far from the operator-confirmed
            # object position is almost certainly robot/table clutter.  Do not
            # let an in-bounds false positive override the confirmed target.
            safe_candidates = [
                candidate
                for candidate in safe_candidates
                if candidate["anchor_error_mm"] <= OPERATOR_ANCHOR_TOLERANCE_MM
            ]
            safe_candidates.sort(
                key=lambda candidate: (
                    candidate["anchor_error_mm"],
                    -candidate["shape_score"],
                )
            )
        else:
            safe_candidates.sort(
                key=lambda candidate: (-candidate["shape_score"], candidate["area"])
            )

        detection = {
            "visible": True,
            "component_count": len(components),
            "safe_component_count": len(safe_candidates),
            "candidates": evaluated,
        }
        if safe_candidates:
            selected = safe_candidates[0]
            # Without an operator anchor, refuse a close ambiguity instead of
            # guessing between two plausible colored objects in the workcell.
            if fallback is None and len(safe_candidates) > 1:
                first = safe_candidates[0]["shape_score"]
                second = safe_candidates[1]["shape_score"]
                if first < second * 1.20:
                    detection.update({"source": "ambiguous_calibrated_camera", "safe": False})
                    self._last_detections[color] = detection
                    return None
            detection.update(selected)
            detection["source"] = "calibrated_camera"
            if fallback is not None:
                detection["selection_anchor_xy_mm"] = [float(fallback[0]), float(fallback[1])]
            self._last_detections[color] = detection
            return tuple(selected["base_xy_mm"])
        if allow_operator_fallback and fallback is not None:
            # The fallback is deliberately explicit and operator supplied.  It
            # is never inferred from an unsafe calibration result, and it has
            # already passed the same TCP/reach checks as normal targets.
            detection.update(
                {
                    "source": "operator_confirmed_fallback",
                    "fallback_xy_mm": [float(fallback[0]), float(fallback[1])],
                    "calibrated_candidates_rejected": True,
                }
            )
            self._last_detections[color] = detection
            return fallback
        detection.update({"source": "rejected_calibrated_camera", "safe": False})
        self._last_detections[color] = detection
        return None

    def _move(self, obs: dict, target: list[float]) -> list[list[float]]:
        return interpolate_tcp(self._current_pose(obs), target)

    @staticmethod
    def _numeric_list(value) -> list[float] | None:
        if value is None:
            return None
        try:
            values = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(values)):
            return None
        return [float(item) for item in values]

    def _write_telemetry(
        self,
        obs: dict,
        action,
        *,
        phase_before: str,
        latency_ms: float,
        error: str | None,
    ) -> None:
        """Record policy evidence without ever changing or stopping robot motion."""

        try:
            action_array = np.asarray(action, dtype=float)
            action_rows = action_array.reshape(-1, 7).tolist()
            images = obs.get("images", {})
            trial = obs.get("trial", {}) or {}
            event = {
                "schema_version": 1,
                "session_id": self._session_id,
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "rollout_index": self._rollout_index,
                "step": self._step,
                "observation_time": obs.get("t"),
                "trial": {
                    key: trial.get(key)
                    for key in ("trial_id", "layout", "instruction", "status")
                    if key in trial
                },
                "robot": {
                    "tcp_pose": self._numeric_list(obs.get("tcp_pose")),
                    "joints": self._numeric_list(obs.get("joints")),
                    "queued_actions": obs.get("queued_actions"),
                    "cell_status": obs.get("status"),
                },
                "cameras": {
                    str(name): {"shape": list(np.asarray(frame).shape)}
                    for name, frame in images.items()
                },
                "policy": {
                    "phase_before": phase_before,
                    "phase_after": self.phase,
                    "result": self.result,
                    "reason": self.reason,
                    "red_target_xy_mm": list(self.red_xy_mm) if self.red_xy_mm else None,
                    "blue_target_xy_mm": list(self.blue_xy_mm) if self.blue_xy_mm else None,
                    "detections": self._last_detections,
                    "wrist_red_visible": self._last_wrist_red_visible,
                    "rl_enabled": False,
                    "operator_fallback_active": self._operator_fallback_active,
                },
                "action": {
                    "format": self.ACTION,
                    "dt_seconds": self.DT,
                    "rows": action_rows,
                    "row_count": len(action_rows),
                    "all_targets_safe": bool(action_rows)
                    and all(tcp_is_safe(row) for row in action_rows),
                },
                "latency_ms": round(float(latency_ms), 3),
                "error": error,
            }
            self._telemetry_dir.mkdir(parents=True, exist_ok=True)
            with self._events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
            self._latest_path.write_text(json.dumps(event, indent=2), encoding="utf-8")
            summary = {
                "schema_version": 1,
                "session_id": self._session_id,
                "rollout_index": self._rollout_index,
                "steps_recorded": self._total_steps,
                "current_phase": self.phase,
                "result": self.result,
                "reason": self.reason,
                "last_latency_ms": event["latency_ms"],
                "last_action_safe": event["action"]["all_targets_safe"],
                "telemetry_error": self._telemetry_error,
            }
            self._summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as telemetry_error:
            # Logging is evidence only. It must never alter a physical action.
            self._telemetry_error = str(telemetry_error)

    def _fail(self, obs: dict, reason: str, *, closed: bool = False) -> list[float]:
        self.phase = "stopped"
        self.result = "stopped"
        self.reason = reason
        return self._hold(obs, closed=closed)

    def act(self, obs: dict):
        phase_before = self.phase
        started = time.perf_counter()
        action = None
        error: str | None = None
        try:
            action = self._act_impl(obs)
            return action
        except Exception as policy_error:
            error = f"{type(policy_error).__name__}: {policy_error}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._step += 1
            self._total_steps += 1
            if action is not None:
                self._write_telemetry(
                    obs,
                    action,
                    phase_before=phase_before,
                    latency_ms=elapsed_ms,
                    error=error,
                )

    def _act_impl(self, obs: dict):
        table = obs.get("images", {}).get("table")
        if self._calibration_error:
            return self._fail(obs, f"calibration unavailable: {self._calibration_error}")
        if table is None:
            return self._fail(obs, "table camera unavailable")

        if self.phase == "locate":
            red = self._table_xy(table, "red")
            blue = self._table_xy(table, "blue")
            if red is None or blue is None:
                return self._hold(obs)
            if math.dist(red, blue) < 30.0:
                return self._fail(obs, "red and blue detections overlap before pickup")
            self.red_xy_mm, self.blue_xy_mm = red, blue
            self.phase = "descend_pick"
            return self._move(obs, self._target(red, SAFE_Z_MM, 0.0))

        if self.red_xy_mm is None or self.blue_xy_mm is None:
            return self._fail(obs, "object positions were not initialized")

        if self.phase == "descend_pick":
            self.phase = "close"
            return self._move(obs, self._target(self.red_xy_mm, PICK_Z_MM, 0.0))
        if self.phase == "close":
            self.phase = "lift"
            return [self._target(self.red_xy_mm, PICK_Z_MM, 1.0)] * 5
        if self.phase == "lift":
            self.phase = "verify_pick"
            return self._move(obs, self._target(self.red_xy_mm, SAFE_Z_MM, 1.0))
        if self.phase == "verify_pick":
            seen = self._table_xy(table, "red", allow_operator_fallback=False)
            wrist_has_red = color_center(obs.get("images", {}).get("wrist", np.zeros((1, 1, 3), np.uint8)), "red") is not None
            self._last_wrist_red_visible = bool(wrist_has_red)
            moved = seen is not None and math.dist(seen, self.red_xy_mm) >= 15.0
            if not wrist_has_red and not moved:
                return self._fail(obs, "pickup could not be verified", closed=True)
            self.phase = "descend_place"
            return self._move(obs, self._target(self.blue_xy_mm, SAFE_Z_MM, 1.0))
        if self.phase == "descend_place":
            self.phase = "release"
            return self._move(obs, self._target(self.blue_xy_mm, PLACE_Z_MM, 1.0))
        if self.phase == "release":
            self.phase = "retreat"
            return [self._target(self.blue_xy_mm, PLACE_Z_MM, 0.0)] * 5
        if self.phase == "retreat":
            self.phase = "verify_place"
            return self._move(obs, self._target(self.blue_xy_mm, SAFE_Z_MM, 0.0))
        if self.phase == "verify_place":
            seen = self._table_xy(table, "red", allow_operator_fallback=False)
            if seen is None or math.dist(seen, self.blue_xy_mm) > 18.0:
                if self._operator_fallback_active:
                    # Calibration cannot independently verify a fallback move.
                    # Finish safely and leave the physical result to the
                    # organizer's camera/scoring rather than claiming success.
                    self.phase = "complete"
                    self.result = "completed_unverified"
                    self.reason = "operator fallback used; final camera verification unavailable"
                    return self._hold(obs)
                return self._fail(obs, "final red-on-blue placement was not verified")
            self.phase = "complete"
            self.result = "success"
            return self._hold(obs)
        return self._hold(obs, closed=self.phase == "stopped" and obs["joints"][-1] >= 0.5)

