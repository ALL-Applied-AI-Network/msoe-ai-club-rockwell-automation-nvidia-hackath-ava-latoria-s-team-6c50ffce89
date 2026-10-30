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
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote

import cv2
import numpy as np


TCP_BOX = np.array([[150.0, 650.0], [-450.0, 450.0], [-334.0, 300.0]])
SAFE_Z_MM = -250.0
PICK_Z_MM = -329.0
PLACE_Z_MM = -319.0
CUBE_HEIGHT_MM = 10.0
MAX_REACH_MM = 550.0
OPERATOR_ANCHOR_TOLERANCE_MM = 75.0
SIX_CUBE_ORDER = ("yellow", "yellow", "blue", "blue", "red", "red")
SIX_CUBE_ENV = "REMOTE_ARM_STACK_ALL_CUBES"
SIX_CUBE_MIN_SEPARATION_MM = 25.0
SIX_CUBE_PLACE_TOLERANCE_MM = 18.0
SIX_CUBE_SOURCE_REFRESH_MM = 20.0
SIX_CUBE_PREFLIGHT_STABLE_FRAMES = 3
SIX_CUBE_PREFLIGHT_MAX_FRAMES = 12
SIX_CUBE_PREFLIGHT_POSITION_TOLERANCE_MM = 5.0
TELEMETRY_ROOT = Path(
    os.environ.get("REMOTE_ARM_TELEMETRY_DIR", "/workspace/team-data/remote-arm-telemetry")
)
FIXED_TARGET_ENV = {
    "red": "REMOTE_ARM_RED_XY_MM",
    "blue": "REMOTE_ARM_BLUE_XY_MM",
}
DEFAULT_INSTRUCTION = "Place the red 10 mm cube on top of the blue 10 mm cube."
SIX_CUBE_INSTRUCTION = (
    "Detect all six 10 mm cubes, group them by color, and build one vertical "
    "tower in this bottom-to-top order: yellow, yellow, blue, blue, red, red. "
    "Verify every grasp and placement before continuing. If detection, "
    "calibration, grasp verification, or placement verification fails, stop "
    "safely rather than guessing."
)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["stack"]},
        "source_color": {"type": "string", "enum": ["red", "blue"]},
        "destination_color": {"type": "string", "enum": ["red", "blue"]},
        "shape": {"type": "string", "enum": ["cube"]},
        "size_mm": {"type": "integer", "enum": [10]},
        "requires_clarification": {"type": "boolean"},
    },
    "required": [
        "action",
        "source_color",
        "destination_color",
        "shape",
        "size_mm",
        "requires_clarification",
    ],
}
SIX_CUBE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["stack_all"]},
        "shape": {"type": "string", "enum": ["cube"]},
        "size_mm": {"type": "integer", "enum": [10]},
        "cube_count": {"type": "integer", "enum": [6]},
        "stack_colors_bottom_to_top": {
            "type": "array",
            "items": {"type": "string", "enum": ["yellow", "blue", "red"]},
            "minItems": 6,
            "maxItems": 6,
        },
        "requires_clarification": {"type": "boolean"},
    },
    "required": [
        "action",
        "shape",
        "size_mm",
        "cube_count",
        "stack_colors_bottom_to_top",
        "requires_clarification",
    ],
}


class InstructionError(ValueError):
    """Raised when a language instruction cannot be converted safely."""


def env_enabled(name: str) -> bool:
    """Read an explicit boolean environment flag and reject typos."""

    value = os.environ.get(name, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def is_six_cube_instruction(instruction: str | None) -> bool:
    """Recognize only clear instructions that request one six-cube tower."""

    normalized = re.sub(r"[^a-z0-9]+", " ", (instruction or "").lower()).strip()
    count_requested = "six cube" in normalized or "6 cube" in normalized
    all_requested = "all cube" in normalized or "all six" in normalized
    stack_requested = "stack" in normalized or "tower" in normalized
    return bool(stack_requested and (count_requested or all_requested))


def _validated_task_plan(raw: dict, *, instruction: str, planner: str) -> dict:
    """Validate model output again before it can select a robot skill."""

    if not isinstance(raw, dict):
        raise InstructionError("instruction planner did not return an object")
    required = {
        "action",
        "source_color",
        "destination_color",
        "shape",
        "size_mm",
        "requires_clarification",
    }
    if set(raw) != required:
        raise InstructionError("instruction plan has unexpected or missing fields")
    if raw["action"] != "stack" or raw["shape"] != "cube":
        raise InstructionError("only cube stacking is currently allowlisted")
    if raw["source_color"] not in FIXED_TARGET_ENV or raw["destination_color"] not in FIXED_TARGET_ENV:
        raise InstructionError("instruction uses an unsupported cube color")
    if raw["source_color"] == raw["destination_color"]:
        raise InstructionError("source and destination cubes must be different")
    if type(raw["size_mm"]) is not int or raw["size_mm"] != 10:
        raise InstructionError("only 10 mm cubes are currently allowlisted")
    if type(raw["requires_clarification"]) is not bool:
        raise InstructionError("requires_clarification must be boolean")
    if raw["requires_clarification"]:
        raise InstructionError("instruction is ambiguous and requires clarification")
    return {
        **raw,
        "instruction": instruction,
        "planner": planner,
    }


def _validated_six_cube_task_plan(
    raw: dict, *, instruction: str, planner: str
) -> dict:
    """Validate a language plan before it can select SafeSixCubeStackV1."""

    if not isinstance(raw, dict):
        raise InstructionError("six-cube planner did not return an object")
    required = {
        "action",
        "shape",
        "size_mm",
        "cube_count",
        "stack_colors_bottom_to_top",
        "requires_clarification",
    }
    if set(raw) != required:
        raise InstructionError("six-cube plan has unexpected or missing fields")
    if raw["action"] != "stack_all" or raw["shape"] != "cube":
        raise InstructionError("six-cube plan must select the stack_all cube skill")
    if type(raw["size_mm"]) is not int or raw["size_mm"] != 10:
        raise InstructionError("six-cube skill requires 10 mm cubes")
    if type(raw["cube_count"]) is not int or raw["cube_count"] != 6:
        raise InstructionError("six-cube skill requires exactly six cubes")
    colors = raw["stack_colors_bottom_to_top"]
    if not isinstance(colors, list) or tuple(colors) != SIX_CUBE_ORDER:
        raise InstructionError(
            "six-cube stack order must be yellow, yellow, blue, blue, red, red"
        )
    if type(raw["requires_clarification"]) is not bool:
        raise InstructionError("requires_clarification must be boolean")
    if raw["requires_clarification"]:
        raise InstructionError("six-cube instruction is ambiguous and requires clarification")
    return {
        **raw,
        "instruction": instruction,
        "planner": planner,
        "execution_skill": "SafeSixCubeStackV1",
    }


def _fixed_six_cube_plan(instruction: str, *, planner: str) -> dict:
    return _validated_six_cube_task_plan(
        {
            "action": "stack_all",
            "shape": "cube",
            "size_mm": 10,
            "cube_count": 6,
            "stack_colors_bottom_to_top": list(SIX_CUBE_ORDER),
            "requires_clarification": False,
        },
        instruction=instruction,
        planner=planner,
    )


def _local_task_plan(instruction: str) -> dict | None:
    """Handle the known benchmark wording without network access or latency."""

    normalized = re.sub(r"[^a-z0-9]+", " ", instruction.lower()).strip()
    verbs_ok = "place" in normalized or "stack" in normalized
    relation_ok = "on top of" in normalized or "onto" in normalized or "on the blue" in normalized
    red_index = normalized.find("red")
    blue_index = normalized.find("blue")
    if (
        verbs_ok
        and relation_ok
        and "cube" in normalized
        and ("10 mm" in normalized or "10mm" in normalized)
        and 0 <= red_index < blue_index
    ):
        return _validated_task_plan(
            {
                "action": "stack",
                "source_color": "red",
                "destination_color": "blue",
                "shape": "cube",
                "size_mm": 10,
                "requires_clarification": False,
            },
            instruction=instruction,
            planner="deterministic",
        )
    return None


def _gemini_task_plan(instruction: str, *, six_cube: bool = False) -> dict:
    """Use Gemini structured output as a planner, never as a motion controller."""

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise InstructionError("unsupported wording and GEMINI_API_KEY is not configured")
    model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if not model or not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise InstructionError("GEMINI_MODEL contains invalid characters")
    try:
        timeout = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "5"))
    except ValueError as exc:
        raise InstructionError("GEMINI_TIMEOUT_SECONDS must be numeric") from exc
    timeout = min(15.0, max(1.0, timeout))
    schema = SIX_CUBE_TASK_SCHEMA if six_cube else TASK_SCHEMA
    skill_note = (
        "The only selectable skill stacks exactly two yellow, two blue, and two red "
        "10 mm cubes in yellow, yellow, blue, blue, red, red order. "
        if six_cube
        else "The only selectable skill stacks one named 10 mm cube on another. "
    )
    prompt = (
        "Convert the robot instruction below into the provided task schema. "
        + skill_note
        + "Do not invent objects, colors, sizes, coordinates, or actions. Mark "
        "requires_clarification true if any required detail is absent or ambiguous. "
        "The output selects a pre-validated skill only; it must never contain "
        "joint angles, poses, code, or motion commands.\n\nInstruction: "
        + instruction
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseFormat": {
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": schema,
                }
            },
        },
    }
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + quote(model, safe="")
        + ":generateContent"
    )
    api_request = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(api_request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        raise InstructionError(f"Gemini request failed with HTTP {exc.code}") from exc
    except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise InstructionError(f"Gemini request failed: {type(exc).__name__}") from exc
    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        raw = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise InstructionError("Gemini response did not contain a valid task plan") from exc
    validator = _validated_six_cube_task_plan if six_cube else _validated_task_plan
    return validator(raw, instruction=instruction, planner="gemini")


def resolve_task_plan(instruction: str | None, *, force_gemini: bool = False) -> dict:
    """Resolve one instruction, preferring the deterministic safe fast path."""

    text = (instruction or DEFAULT_INSTRUCTION).strip()
    if not text:
        raise InstructionError("task instruction is empty")
    if is_six_cube_instruction(text):
        return (
            _gemini_task_plan(text, six_cube=True)
            if force_gemini
            else _fixed_six_cube_plan(text, planner="deterministic")
        )
    local = _local_task_plan(text)
    if local is not None and not force_gemini:
        return local
    return _gemini_task_plan(text)


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
    elif color == "yellow":
        mask = (
            (r > 90)
            & (g > 75)
            & (r > b * 1.35)
            & (g > b * 1.35)
            & (np.abs(r - g) < 110)
        )
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
    """Execute an allowlisted language task through the deterministic state machine."""

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
        self._red_seen_at_pick_target = False
        self._pick_verification_source: str | None = None
        self._last_commanded_target: list[float] | None = None
        self.task_plan: dict | None = None
        self._six_cube_mode = env_enabled(SIX_CUBE_ENV)
        self._six_cube_plan: list[dict] = []
        self._six_cube_index = 0
        self._six_cube_verified_layers: list[dict] = []
        self._six_current: dict | None = None
        self._six_source_seen_at_pick_target = False
        self._six_pick_verification_source: str | None = None
        self._six_tower_xy_mm: tuple[float, float] | None = None
        self._six_preflight_plan: list[dict] | None = None
        self._six_preflight_stable_frames = 0
        self._six_preflight_attempts = 0
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

    def _target_reached(
        self,
        obs: dict,
        target: list[float],
        *,
        position_tolerance_mm: float = 2.0,
    ) -> bool:
        current = self._current_pose(obs)
        if current is None:
            return False
        return math.dist(current[:3], target[:3]) <= position_tolerance_mm

    @staticmethod
    def _gripper_reached(obs: dict, *, closed: bool) -> bool:
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
        if not joints.size or not np.isfinite(joints[-1]):
            return False
        return bool(joints[-1] >= 0.5) == closed

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

    def _table_xys(self, image: np.ndarray, color: str) -> list[tuple[float, float]]:
        """Return every safely calibrated component for six-cube planning.

        Unlike the proven one-cube path, this method never guesses between
        duplicate colors and never uses the single-target operator fallback.
        The six-cube planner validates the exact expected component count.
        """

        key = f"six_cube_{color}"
        if self._cams is None or self._pixel_to_base is None:
            self._last_detections[key] = {
                "visible": False,
                "reason": "calibration_unavailable",
            }
            return []
        components = color_components(image, color)
        native_w, native_h = self._cams["table"]["size"]
        height, width = image.shape[:2]
        evaluated: list[dict] = []
        safe_xy: list[tuple[float, float]] = []
        for component in components:
            center = component["center"]
            uv_native = (center[0] * native_w / width, center[1] * native_h / height)
            xyz_m = self._pixel_to_base(self._cams["table"], uv_native)
            xy = float(xyz_m[0] * 1000.0), float(xyz_m[1] * 1000.0)
            safe = tcp_is_safe(self._target(xy, SAFE_Z_MM, 0.0)) and tcp_is_safe(
                self._target(xy, PICK_Z_MM, 0.0)
            )
            evaluated.append(
                {
                    "pixel": [float(center[0]), float(center[1])],
                    "native_pixel": [float(uv_native[0]), float(uv_native[1])],
                    "base_xy_mm": [float(xy[0]), float(xy[1])],
                    "safe": bool(safe),
                    "area": int(component["area"]),
                    "bbox": component["bbox"],
                    "shape_score": float(component["shape_score"]),
                }
            )
            if safe:
                safe_xy.append(xy)
        safe_xy.sort(key=lambda xy: (xy[0], xy[1]))
        self._last_detections[key] = {
            "visible": bool(components),
            "component_count": len(components),
            "safe_component_count": len(safe_xy),
            "candidates": evaluated,
            "source": "calibrated_camera_all_components",
        }
        return safe_xy

    def _build_six_cube_plan(self, table: np.ndarray) -> list[dict]:
        """Freeze a deterministic five-move plan that builds a six-cube tower."""

        required = {"yellow": 2, "blue": 2, "red": 2}
        detected: dict[str, list[tuple[float, float]]] = {}
        for color, expected in required.items():
            detected[color] = self._table_xys(table, color)
            details = self._last_detections.get(f"six_cube_{color}", {})
            if (
                len(detected[color]) != expected
                or details.get("component_count") != expected
                or details.get("safe_component_count") != expected
            ):
                raise ValueError(
                    f"six-cube mode requires exactly {expected} safe {color} cubes; "
                    f"found {len(detected[color])}"
                )

        points = [xy for color in ("yellow", "blue", "red") for xy in detected[color]]
        for left in range(len(points)):
            for right in range(left + 1, len(points)):
                if math.dist(points[left], points[right]) < SIX_CUBE_MIN_SEPARATION_MM:
                    raise ValueError(
                        "six-cube starting positions must be at least "
                        f"{SIX_CUBE_MIN_SEPARATION_MM:.0f} mm apart"
                    )

        # Leave the yellow cube with the most clearance in place as the tower
        # base. Five verified pick/place moves then produce a six-cube tower.
        yellow = detected["yellow"]
        base_index = max(
            range(len(yellow)),
            key=lambda index: min(
                math.dist(yellow[index], point)
                for point in points
                if point != yellow[index]
            ),
        )
        tower_xy = yellow[base_index]
        by_color = {
            "yellow": [xy for index, xy in enumerate(yellow) if index != base_index],
            "blue": list(detected["blue"]),
            "red": list(detected["red"]),
        }
        for candidates in by_color.values():
            candidates.sort(key=lambda xy: (math.dist(xy, tower_xy), xy[0], xy[1]))

        plan: list[dict] = []
        for layer, color in enumerate(SIX_CUBE_ORDER[1:], start=1):
            source_xy = by_color[color].pop(0)
            place_z = PICK_Z_MM + CUBE_HEIGHT_MM * layer
            pick_target = self._target(source_xy, PICK_Z_MM, 0.0)
            place_target = self._target(tower_xy, place_z, 1.0)
            if not tcp_is_safe(pick_target) or not tcp_is_safe(place_target):
                raise ValueError(f"unsafe six-cube target at layer {layer}")
            if SAFE_Z_MM - place_z < 20.0:
                raise ValueError(f"insufficient tower clearance at layer {layer}")
            plan.append(
                {
                    "layer": layer,
                    "color": color,
                    "source_xy_mm": source_xy,
                    "tower_xy_mm": tower_xy,
                    "place_z_mm": place_z,
                }
            )
        self._six_tower_xy_mm = tower_xy
        return plan

    def _start_six_cube_pick(self, index: int) -> None:
        self._six_cube_index = index
        self._six_current = self._six_cube_plan[index]
        self._six_source_seen_at_pick_target = True
        self._six_pick_verification_source = None

    @staticmethod
    def _six_cube_plans_match(left: list[dict], right: list[dict]) -> bool:
        """Require stable calibrated detections before physical motion starts."""

        if len(left) != len(right):
            return False
        for previous, current in zip(left, right):
            if previous["layer"] != current["layer"] or previous["color"] != current["color"]:
                return False
            if (
                math.dist(previous["source_xy_mm"], current["source_xy_mm"])
                > SIX_CUBE_PREFLIGHT_POSITION_TOLERANCE_MM
                or math.dist(previous["tower_xy_mm"], current["tower_xy_mm"])
                > SIX_CUBE_PREFLIGHT_POSITION_TOLERANCE_MM
            ):
                return False
        return True

    def _refresh_six_cube_source(self, table: np.ndarray) -> None:
        """Update the next source from the latest frame without changing identity."""

        if self._six_current is None or self._six_tower_xy_mm is None:
            raise ValueError("six-cube plan was not initialized")
        color = str(self._six_current["color"])
        planned = tuple(self._six_current["source_xy_mm"])
        candidates = [
            xy
            for xy in self._table_xys(table, color)
            if math.dist(xy, self._six_tower_xy_mm) > SIX_CUBE_PLACE_TOLERANCE_MM
        ]
        if not candidates:
            raise ValueError(f"next {color} source cube is not visible")
        selected = min(candidates, key=lambda xy: math.dist(xy, planned))
        error = math.dist(selected, planned)
        if error > SIX_CUBE_SOURCE_REFRESH_MM:
            raise ValueError(
                f"next {color} source moved {error:.1f} mm from its planned location"
            )
        self._six_current["source_xy_mm"] = selected

    def _act_six_cube(self, obs: dict, table: np.ndarray, instruction: str | None):
        """Execute a calibrated, supervised six-cube tower state machine."""

        if self._operator_fallback_active:
            return self._fail(
                obs,
                "six-cube mode requires calibrated camera positions; operator fallback is unsupported",
            )
        if self.phase == "locate":
            if self.task_plan is None:
                try:
                    if is_six_cube_instruction(instruction):
                        self.task_plan = resolve_task_plan(instruction)
                    else:
                        self.task_plan = _fixed_six_cube_plan(
                            SIX_CUBE_INSTRUCTION,
                            planner="operator_env",
                        )
                except InstructionError as instruction_error:
                    return self._fail(obs, f"six-cube instruction rejected: {instruction_error}")
            if self.task_plan.get("execution_skill") != "SafeSixCubeStackV1":
                return self._fail(obs, "six-cube plan selected an unsupported execution skill")
            self._six_preflight_attempts += 1
            try:
                candidate_plan = self._build_six_cube_plan(table)
            except ValueError as planning_error:
                self._six_preflight_plan = None
                self._six_preflight_stable_frames = 0
                self.reason = f"six-cube preflight waiting: {planning_error}"
                if self._six_preflight_attempts >= SIX_CUBE_PREFLIGHT_MAX_FRAMES:
                    return self._fail(obs, f"six-cube planning failed: {planning_error}")
                return self._hold(obs)
            if self._six_preflight_plan is not None and self._six_cube_plans_match(
                self._six_preflight_plan, candidate_plan
            ):
                self._six_preflight_stable_frames += 1
            else:
                self._six_preflight_plan = candidate_plan
                self._six_preflight_stable_frames = 1
            if self._six_preflight_stable_frames < SIX_CUBE_PREFLIGHT_STABLE_FRAMES:
                self.reason = (
                    "six-cube preflight confirming stable detections "
                    f"{self._six_preflight_stable_frames}/"
                    f"{SIX_CUBE_PREFLIGHT_STABLE_FRAMES}"
                )
                return self._hold(obs)
            self._six_cube_plan = candidate_plan
            self.reason = None
            self.task_plan["tower_xy_mm"] = list(self._six_tower_xy_mm)
            self._start_six_cube_pick(0)
            self.phase = "descend_pick"
            source_xy = tuple(self._six_current["source_xy_mm"])
            return self._move(obs, self._target(source_xy, SAFE_Z_MM, 0.0))

        current = self._six_current
        if current is None or self._six_tower_xy_mm is None:
            return self._fail(obs, "six-cube plan was not initialized")
        source_xy = tuple(current["source_xy_mm"])
        color = str(current["color"])
        place_z = float(current["place_z_mm"])

        if self.phase == "descend_pick":
            above_pick = self._target(source_xy, SAFE_Z_MM, 0.0)
            if not self._target_reached(obs, above_pick):
                return self._move(obs, above_pick)
            self.phase = "close"
            return self._move(obs, self._target(source_xy, PICK_Z_MM, 0.0))
        if self.phase == "close":
            pick_open = self._target(source_xy, PICK_Z_MM, 0.0)
            if not self._target_reached(obs, pick_open):
                return self._move(obs, pick_open)
            self.phase = "lift"
            return self._repeat_target(self._target(source_xy, PICK_Z_MM, 1.0), 5)
        if self.phase == "lift":
            pick_closed = self._target(source_xy, PICK_Z_MM, 1.0)
            if not self._target_reached(obs, pick_closed) or not self._gripper_reached(
                obs, closed=True
            ):
                return self._move(obs, pick_closed)
            self.phase = "verify_pick"
            return self._move(obs, self._target(source_xy, SAFE_Z_MM, 1.0))
        if self.phase == "verify_pick":
            lifted = self._target(source_xy, SAFE_Z_MM, 1.0)
            if not self._target_reached(obs, lifted) or not self._gripper_reached(
                obs, closed=True
            ):
                return self._move(obs, lifted)
            remaining = self._table_xys(table, color)
            source_still_present = any(
                math.dist(xy, source_xy) < 15.0 for xy in remaining
            )
            wrist_has_cube = color_center(
                obs.get("images", {}).get(
                    "wrist", np.zeros((1, 1, 3), dtype=np.uint8)
                ),
                color,
            ) is not None
            joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
            gripper_closed = bool(
                joints.size and np.isfinite(joints[-1]) and joints[-1] >= 0.5
            )
            if wrist_has_cube:
                self._six_pick_verification_source = "wrist_camera"
            elif (
                self._six_source_seen_at_pick_target
                and not source_still_present
                and gripper_closed
            ):
                self._six_pick_verification_source = (
                    "table_source_disappearance_with_closed_gripper"
                )
            else:
                return self._fail(
                    obs,
                    f"layer {current['layer']} {color} pickup could not be verified",
                    closed=True,
                )
            self.phase = "descend_place"
            return self._move(
                obs, self._target(self._six_tower_xy_mm, SAFE_Z_MM, 1.0)
            )
        if self.phase == "descend_place":
            above_place = self._target(self._six_tower_xy_mm, SAFE_Z_MM, 1.0)
            if not self._target_reached(obs, above_place):
                return self._move(obs, above_place)
            self.phase = "release"
            return self._move(
                obs, self._target(self._six_tower_xy_mm, place_z, 1.0)
            )
        if self.phase == "release":
            place_closed = self._target(self._six_tower_xy_mm, place_z, 1.0)
            if not self._target_reached(obs, place_closed):
                return self._move(obs, place_closed)
            self.phase = "retreat"
            return self._repeat_target(
                self._target(self._six_tower_xy_mm, place_z, 0.0), 5
            )
        if self.phase == "retreat":
            place_open = self._target(self._six_tower_xy_mm, place_z, 0.0)
            if not self._target_reached(obs, place_open) or not self._gripper_reached(
                obs, closed=False
            ):
                return self._move(obs, place_open)
            self.phase = "verify_place"
            return self._move(
                obs, self._target(self._six_tower_xy_mm, SAFE_Z_MM, 0.0)
            )
        if self.phase == "verify_place":
            retreated = self._target(self._six_tower_xy_mm, SAFE_Z_MM, 0.0)
            if not self._target_reached(obs, retreated):
                return self._move(obs, retreated)
            placed = any(
                math.dist(xy, self._six_tower_xy_mm)
                <= SIX_CUBE_PLACE_TOLERANCE_MM
                for xy in self._table_xys(table, color)
            )
            if not placed:
                return self._fail(
                    obs,
                    f"layer {current['layer']} {color} placement was not verified",
                )
            self._six_cube_verified_layers.append(
                {
                    "layer": int(current["layer"]),
                    "color": color,
                    "place_z_mm": place_z,
                    "pick_verification_source": self._six_pick_verification_source,
                }
            )
            next_index = self._six_cube_index + 1
            if next_index >= len(self._six_cube_plan):
                self.phase = "complete"
                self.result = "success"
                self.reason = "six-cube tower completed with five verified placements"
                return self._hold(obs)
            self._start_six_cube_pick(next_index)
            try:
                self._refresh_six_cube_source(table)
            except ValueError as source_error:
                return self._fail(obs, f"six-cube source refresh failed: {source_error}")
            self.phase = "descend_pick"
            next_xy = tuple(self._six_current["source_xy_mm"])
            return self._move(obs, self._target(next_xy, SAFE_Z_MM, 0.0))
        return self._hold(
            obs,
            closed=self.phase == "stopped"
            and bool(np.asarray(obs.get("joints", [0]))[-1] >= 0.5),
        )

    def _move(self, obs: dict, target: list[float]) -> list[list[float]]:
        # The physical observation trails the queued command.  Replanning each
        # chunk from that delayed pose made the arm pull back toward an older Z
        # value every time the client asked for more actions.  Continue from
        # the last commanded endpoint instead so consecutive chunks remain
        # monotonic, then use _target_reached() to gate the next phase.
        start = (
            self._last_commanded_target[:6]
            if self._last_commanded_target is not None
            else self._current_pose(obs)
        )
        rows = interpolate_tcp(start, target)
        self._last_commanded_target = list(rows[-1])
        return rows

    def _repeat_target(self, target: list[float], count: int) -> list[list[float]]:
        self._last_commanded_target = list(target)
        return [list(target) for _ in range(count)]

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
                    "policy_name": (
                        "SafeSixCubeStackV1"
                        if self._six_cube_mode
                        else "SafeRedOnBlueV1"
                    ),
                    "red_target_xy_mm": list(self.red_xy_mm) if self.red_xy_mm else None,
                    "blue_target_xy_mm": list(self.blue_xy_mm) if self.blue_xy_mm else None,
                    "detections": self._last_detections,
                    "wrist_red_visible": self._last_wrist_red_visible,
                    "pick_verification_source": self._pick_verification_source,
                    "task_plan": self.task_plan,
                    "six_cube": {
                        "enabled": self._six_cube_mode,
                        "move_index": self._six_cube_index,
                        "move_count": len(self._six_cube_plan),
                        "tower_xy_mm": (
                            list(self._six_tower_xy_mm)
                            if self._six_tower_xy_mm is not None
                            else None
                        ),
                        "current": self._six_current,
                        "verified_layers": self._six_cube_verified_layers,
                        "pick_verification_source": self._six_pick_verification_source,
                    },
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

        trial = obs.get("trial", {}) or {}
        instruction = (
            trial.get("instruction") if isinstance(trial, dict) else None
        ) or os.environ.get("REMOTE_ARM_INSTRUCTION")
        if self.phase == "locate" and is_six_cube_instruction(instruction):
            self._six_cube_mode = True
        if self._six_cube_mode:
            return self._act_six_cube(obs, table, instruction)

        if self.phase == "locate":
            if self.task_plan is None:
                try:
                    self.task_plan = resolve_task_plan(instruction)
                except InstructionError as instruction_error:
                    return self._fail(obs, f"instruction rejected: {instruction_error}")
                # The planner may only select skills that this physical policy
                # has explicitly implemented and tested. The structured
                # parser is intentionally more general than today's single
                # execution skill so future skills can be added safely.
                supported = (
                    self.task_plan["action"] == "stack"
                    and self.task_plan["source_color"] == "red"
                    and self.task_plan["destination_color"] == "blue"
                    and self.task_plan["shape"] == "cube"
                    and self.task_plan["size_mm"] == 10
                )
                if not supported:
                    return self._fail(
                        obs,
                        "instruction resolved to a task without an allowlisted physical skill",
                    )
            red = self._table_xy(table, "red")
            blue = self._table_xy(table, "blue")
            if red is None or blue is None:
                return self._hold(obs)
            if math.dist(red, blue) < 30.0:
                return self._fail(obs, "red and blue detections overlap before pickup")
            self.red_xy_mm, self.blue_xy_mm = red, blue
            red_detection = self._last_detections.get("red", {})
            self._red_seen_at_pick_target = bool(
                red_detection.get("visible")
                and red_detection.get("source") == "calibrated_camera"
            )
            self.phase = "descend_pick"
            return self._move(obs, self._target(red, SAFE_Z_MM, 0.0))

        if self.red_xy_mm is None or self.blue_xy_mm is None:
            return self._fail(obs, "object positions were not initialized")

        if self.phase == "descend_pick":
            above_pick = self._target(self.red_xy_mm, SAFE_Z_MM, 0.0)
            if not self._target_reached(obs, above_pick):
                return self._move(obs, above_pick)
            self.phase = "close"
            return self._move(obs, self._target(self.red_xy_mm, PICK_Z_MM, 0.0))
        if self.phase == "close":
            pick_open = self._target(self.red_xy_mm, PICK_Z_MM, 0.0)
            if not self._target_reached(obs, pick_open):
                return self._move(obs, pick_open)
            self.phase = "lift"
            return self._repeat_target(
                self._target(self.red_xy_mm, PICK_Z_MM, 1.0),
                5,
            )
        if self.phase == "lift":
            pick_closed = self._target(self.red_xy_mm, PICK_Z_MM, 1.0)
            if not self._target_reached(obs, pick_closed) or not self._gripper_reached(
                obs, closed=True
            ):
                return self._move(obs, pick_closed)
            self.phase = "verify_pick"
            return self._move(obs, self._target(self.red_xy_mm, SAFE_Z_MM, 1.0))
        if self.phase == "verify_pick":
            lifted = self._target(self.red_xy_mm, SAFE_Z_MM, 1.0)
            if not self._target_reached(obs, lifted) or not self._gripper_reached(
                obs, closed=True
            ):
                return self._move(obs, lifted)
            seen = self._table_xy(table, "red", allow_operator_fallback=False)
            wrist_has_red = color_center(obs.get("images", {}).get("wrist", np.zeros((1, 1, 3), np.uint8)), "red") is not None
            self._last_wrist_red_visible = bool(wrist_has_red)
            moved = seen is not None and math.dist(seen, self.red_xy_mm) >= 15.0
            joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1)
            gripper_closed = bool(joints.size and np.isfinite(joints[-1]) and joints[-1] >= 0.5)
            disappeared_after_lift = bool(
                self._red_seen_at_pick_target and seen is None and gripper_closed
            )
            if wrist_has_red:
                self._pick_verification_source = "wrist_camera"
            elif moved:
                self._pick_verification_source = "table_camera_moved"
            elif disappeared_after_lift:
                # On the physical cell the 10 mm cube is normally hidden by
                # the closed gripper, so the wrist camera cannot reliably see
                # it.  A calibrated pre-pick observation followed by table
                # disappearance, a closed gripper, and a completed lift is
                # sufficient evidence to continue.  Operator-only fallback
                # targets do not qualify for this path.
                self._pick_verification_source = "table_disappearance_with_closed_gripper"
            else:
                return self._fail(obs, "pickup could not be verified", closed=True)
            self.phase = "descend_place"
            return self._move(obs, self._target(self.blue_xy_mm, SAFE_Z_MM, 1.0))
        if self.phase == "descend_place":
            above_place = self._target(self.blue_xy_mm, SAFE_Z_MM, 1.0)
            if not self._target_reached(obs, above_place):
                return self._move(obs, above_place)
            self.phase = "release"
            return self._move(obs, self._target(self.blue_xy_mm, PLACE_Z_MM, 1.0))
        if self.phase == "release":
            place_closed = self._target(self.blue_xy_mm, PLACE_Z_MM, 1.0)
            if not self._target_reached(obs, place_closed):
                return self._move(obs, place_closed)
            self.phase = "retreat"
            return self._repeat_target(
                self._target(self.blue_xy_mm, PLACE_Z_MM, 0.0),
                5,
            )
        if self.phase == "retreat":
            place_open = self._target(self.blue_xy_mm, PLACE_Z_MM, 0.0)
            if not self._target_reached(obs, place_open) or not self._gripper_reached(
                obs, closed=False
            ):
                return self._move(obs, place_open)
            self.phase = "verify_place"
            return self._move(obs, self._target(self.blue_xy_mm, SAFE_Z_MM, 0.0))
        if self.phase == "verify_place":
            retreated = self._target(self.blue_xy_mm, SAFE_Z_MM, 0.0)
            if not self._target_reached(obs, retreated):
                return self._move(obs, retreated)
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

