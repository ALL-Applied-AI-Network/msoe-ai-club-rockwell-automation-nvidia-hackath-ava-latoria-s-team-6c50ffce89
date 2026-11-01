"""Dependency-free evidence bridge for the demo dashboard.

The simulator owns the run directory and writes files atomically.  This module
never starts, stops, or controls the robot. Its only write is a bounded human
review label appended to ``dashboard_feedback.jsonl`` for later offline use.
"""

from __future__ import annotations

import json
import importlib.util
import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlsplit


JSON_ENDPOINTS = {
    "/api/status": "live_status.json",
    "/api/metrics": "end_to_end_metrics.json",
}
SERVABLE_SUFFIXES = {
    ".json",
    ".md",
    ".csv",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".mp4",
    ".webm",
}
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm"}
KEY_ARTIFACT_NAMES = {
    "live_status.json",
    "end_to_end_metrics.json",
    "end_to_end_metrics.md",
    "execution_events.json",
    "execution_report.json",
}
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
_FEEDBACK_LABELS = {"GOOD", "BAD"}
_FEEDBACK_KINDS = {"grasp", "stack"}
_MAX_FEEDBACK_BYTES = 64 * 1024
_POLICY_PLANNER: Any = None


def _planning_values(value: Any) -> dict[str, Any]:
    """Expose only the structured, validated task-plan fields used by the UI."""
    plan = value if isinstance(value, dict) else {}
    action = plan.get("action")
    source = plan.get("source_color")
    destination = plan.get("destination_color")
    execution_skill = plan.get("execution_skill")
    requires_clarification = bool(plan.get("requires_clarification"))
    ready = bool(execution_skill) or all((action, source, destination))
    status = (
        "waiting"
        if not plan
        else "blocked"
        if requires_clarification or not ready
        else "ready"
    )
    summary = plan.get("summary")
    if summary is None and execution_skill:
        order = plan.get("tower_order")
        summary = (
            f"{execution_skill}: {' → '.join(str(item) for item in order)}"
            if isinstance(order, list) and order
            else str(execution_skill)
        )
    if summary is None and action and source and destination:
        size_mm = plan.get("size_mm")
        shape = plan.get("shape")
        object_name = " ".join(
            str(item)
            for item in (size_mm, "mm" if size_mm is not None else None, shape)
            if item is not None
        )
        summary = f"{str(action).capitalize()} {source} {object_name} on {destination}".strip()
    return {
        "status": status,
        "planner": plan.get("planner"),
        "instruction": plan.get("instruction"),
        "action": action,
        "source_color": source,
        "destination_color": destination,
        "shape": plan.get("shape"),
        "size_mm": plan.get("size_mm"),
        "execution_skill": execution_skill,
        "tower_order": plan.get("tower_order"),
        "requires_clarification": requires_clarification,
        "summary": summary,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    """Return one JSON object, or an empty object while a producer is writing."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _feedback_rows(run_dir: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with (run_dir / "dashboard_feedback.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except (FileNotFoundError, OSError):
        return []
    return rows[-limit:]


def _validated_feedback(value: Any, *, run_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("feedback must be a JSON object")
    kind = str(value.get("kind") or "").lower()
    label = str(value.get("label") or "").upper()
    event_id = str(value.get("event_id") or "").strip()
    if kind not in _FEEDBACK_KINDS:
        raise ValueError("kind must be grasp or stack")
    if label not in _FEEDBACK_LABELS:
        raise ValueError("label must be GOOD or BAD")
    if not event_id or len(event_id) > 160:
        raise ValueError("event_id is required and must be 160 characters or fewer")
    submitted_run = str(value.get("run_id") or run_id)
    if submitted_run != run_id:
        raise ValueError("run_id does not match the active run")
    confidence = value.get("model_confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise ValueError("model_confidence must be numeric or null")
    notes = str(value.get("notes") or "").strip()
    if len(notes) > 500:
        raise ValueError("notes must be 500 characters or fewer")
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": "dashboard_human_review",
        "run_id": run_id,
        "event_id": event_id,
        "kind": kind,
        "label": label,
        "training_label": (
            "GOOD" if label == "GOOD" else "SHIFTED"
            if kind == "grasp"
            else "PLACEMENT_FAILURE"
        ),
        "stage": value.get("stage"),
        "grab": value.get("grab"),
        "model_prediction": value.get("model_prediction"),
        "model_confidence": confidence,
        "notes": notes or None,
    }


def _append_feedback(run_dir: Path, feedback: dict[str, Any]) -> None:
    with (run_dir / "dashboard_feedback.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(feedback, separators=(",", ":")) + "\n")


def _instruction_plan(value: Any, *, run_dir: Path) -> dict[str, Any]:
    """Create a dry-run plan through the same allowlisted planner as the robot policy."""
    if not isinstance(value, dict):
        raise ValueError("planning request must be a JSON object")
    instruction = str(value.get("instruction") or "").strip()
    if not instruction or len(instruction) > 1000:
        raise ValueError("instruction is required and must be 1000 characters or fewer")
    normalized = re.sub(r"[^a-z0-9]+", " ", instruction.lower()).strip()
    six_cube = (
        ("stack" in normalized or "tower" in normalized)
        and ("six cube" in normalized or "6 cube" in normalized or "all six" in normalized)
    )
    if six_cube:
        plan = {
            "action": "stack_all",
            "shape": "cube",
            "size_mm": 10,
            "cube_count": 6,
            "stack_colors_bottom_to_top": [
                "yellow", "yellow", "blue", "blue", "red", "red"
            ],
            "requires_clarification": False,
            "instruction": instruction,
            "planner": "deterministic",
            "execution_skill": "SafeSixCubeStackV1",
            "tower_order": ["yellow", "yellow", "blue", "blue", "red", "red"],
        }
    elif (
        ("place" in normalized or "stack" in normalized)
        and ("on top of" in normalized or "onto" in normalized or "on the blue" in normalized)
        and "cube" in normalized
        and ("10 mm" in normalized or "10mm" in normalized)
        and 0 <= normalized.find("red") < normalized.find("blue")
    ):
        plan = {
            "action": "stack",
            "source_color": "red",
            "destination_color": "blue",
            "shape": "cube",
            "size_mm": 10,
            "requires_clarification": False,
            "instruction": instruction,
            "planner": "deterministic",
        }
    else:
        global _POLICY_PLANNER
        if _POLICY_PLANNER is None:
            policy_path = Path(__file__).resolve().parents[2] / "hardware" / "remote_cell" / "policy.py"
            spec = importlib.util.spec_from_file_location("dashboard_remote_policy", policy_path)
            if spec is None or spec.loader is None:
                raise ValueError("validated instruction planner is unavailable")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _POLICY_PLANNER = module
        try:
            plan = _POLICY_PLANNER.resolve_task_plan(instruction)
        except _POLICY_PLANNER.InstructionError as error:
            raise ValueError(str(error)) from error
    sessions = _session_manifest(run_dir)["sessions"]
    matching_skill = plan.get("execution_skill") or plan.get("action")
    successful_sessions = sum(1 for session in sessions if session.get("result") == "success")
    return {
        "mode": "dry_run",
        "planning": _planning_values(plan),
        "plan": plan,
        "evidence": {
            "matching_skill": matching_skill,
            "nearby_sessions": len(sessions),
            "successful_sessions": successful_sessions,
            "human_feedback_labels": len(_feedback_rows(run_dir)),
        },
        "safety_boundary": (
            "Plan preview only. Execution still requires state-machine validation, "
            "workspace checks, and supervised robot startup."
        ),
    }


def _video_file(run_dir: Path) -> Path | None:
    """Choose the most useful recorded run video without hard-coding its name."""
    videos: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".webm"}:
            continue
        try:
            path.resolve().relative_to(run_dir)
        except ValueError:
            continue
        videos.append(path)

    def priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        preferred = 0 if "sort_then_stack" in name else 1 if "demo" in name else 2
        return preferred, -path.stat().st_mtime_ns, path.as_posix()

    return min(videos, key=priority, default=None)


def _physical_dashboard(run_dir: Path, latest: dict[str, Any]) -> dict[str, Any]:
    """Normalize the remote-cell evidence log into the dashboard contract."""
    summary = _read_json_object(run_dir / "summary.json")
    policy = latest.get("policy") if isinstance(latest.get("policy"), dict) else {}
    action = latest.get("action") if isinstance(latest.get("action"), dict) else {}
    result = str(policy.get("result") or summary.get("result") or "pending")
    phase = str(policy.get("phase_after") or summary.get("current_phase") or "starting")
    state = (
        "failed"
        if result in {"failed", "stopped"}
        else "completed"
        if result == "success"
        else "running"
    )
    row_count = int(action.get("row_count") or 0)
    selected_action = f"TCP motion chunk ({row_count} targets)" if row_count else None
    verified = True if result == "success" else False if state == "failed" else None
    detections = policy.get("detections") if isinstance(policy.get("detections"), dict) else {}
    six_cube = policy.get("six_cube") if isinstance(policy.get("six_cube"), dict) else {}
    task_plan = policy.get("task_plan") if isinstance(policy.get("task_plan"), dict) else {}
    cameras = latest.get("cameras") if isinstance(latest.get("cameras"), dict) else {}
    return {
        "source": "physical_robot",
        "state": state,
        "phase": phase,
        "mode": "remote",
        "updated_at": latest.get("wall_time_utc"),
        "current": {
            "stage": "physical",
            "grab": int(latest.get("rollout_index") or 0) + 1,
            "predicted_class": None,
            "confidence": None,
            "pickup_verified": verified,
            "placement_verified": verified,
            "placement_error_mm": None,
            "cycle_time_s": None,
            "state_machine_step": phase,
            "selected_action": selected_action,
            "rl": {"action": None, "actions": [], "safe_stop": state == "failed"},
            "policy_name": policy.get("policy_name"),
            "reason": policy.get("reason"),
            "detections": detections,
            "detection_count": len(detections),
            "task_plan": task_plan,
            "six_cube": six_cube,
            "latency_ms": latest.get("latency_ms"),
            "action_safe": action.get("all_targets_safe"),
            "action_target_count": row_count,
            "camera_names": sorted(str(name) for name in cameras),
            "model_status": "not_evaluated",
            "model_reason": (
                "The remote policy did not emit a grasp-classifier score for this step"
            ),
        },
        "progress": {
            "completed_grabs": int(summary.get("rollout_index") or 0),
            "total_grabs": None,
            "percent": None,
        },
        "metrics": {
            "steps_recorded": int(summary.get("steps_recorded") or 0),
            "last_latency_ms": summary.get("last_latency_ms"),
            "last_action_safe": summary.get("last_action_safe"),
        },
        "events": [],
        "planning": _planning_values(task_plan),
        "result": result,
        "error": latest.get("error") or summary.get("telemetry_error"),
    }


def _archived_simulation_dashboard(run_dir: Path) -> dict[str, Any] | None:
    """Normalize a completed simulator run that predates live_status.json."""
    archived = _read_json_object(run_dir / "end_to_end_metrics.json")
    events = archived.get("grabs")
    if not isinstance(events, list):
        return None
    valid_events = [event for event in events if isinstance(event, dict)]
    if not valid_events:
        return None
    classifier = archived.get("classifier") if isinstance(archived.get("classifier"), dict) else {}
    completed = len(valid_events)
    scored = int(classifier.get("scored_grabs") or 0)
    verified_placements = sum(
        1 for event in valid_events if event.get("placement_verified") is True
    )
    successful_pickups = sum(
        1 for event in valid_events if event.get("pickup_verified") is True
    )
    state = "completed" if archived.get("full_process_complete") else "failed"
    return {
        "source": "simulation",
        "state": state,
        "phase": str(valid_events[-1].get("stage") or "complete"),
        "mode": "recorded",
        "updated_at": None,
        "current": valid_events[-1],
        "progress": {
            "completed_grabs": completed,
            "total_grabs": completed,
            "percent": 100 if state == "completed" else None,
            "sort_completed": sum(1 for event in valid_events if event.get("stage") == "sort"),
            "stack_completed": sum(1 for event in valid_events if event.get("stage") == "stack"),
        },
        "metrics": {
            "successful_pickups": successful_pickups,
            "verified_placements": verified_placements,
            "classifier_scored": scored,
            "classifier_correct": int(classifier.get("correct") or 0),
            "classifier_accuracy": classifier.get("accuracy"),
            "classifier_coverage": scored / completed if completed else None,
        },
        "events": valid_events,
        "planning": _planning_values(archived.get("planning") or archived.get("task_plan")),
        "result": "success" if state == "completed" else "failed",
        "error": None,
    }


def _display_values(
    current: dict[str, Any], phase: str, overall_success_rate: float | None
) -> dict[str, Any]:
    rl = current.get("rl") if isinstance(current.get("rl"), dict) else {}
    rl_action = rl.get("action")
    pickup = current.get("pickup_verified")
    grasp_result = "pending" if pickup is None else "verified" if pickup else "failed"
    state_machine_step = current.get("state_machine_step")
    if not state_machine_step:
        if current.get("placement_verified"):
            state_machine_step = "verify placement"
        elif pickup:
            state_machine_step = "carry"
        elif current:
            state_machine_step = "verify grasp"
        else:
            state_machine_step = phase
    return {
        "phase": phase,
        "prediction": current.get("predicted_class"),
        "confidence": current.get("confidence"),
        "state_machine_step": state_machine_step,
        "rl_correction": rl_action,
        "selected_action": current.get("selected_action")
        or rl_action
        or current.get("visual_gate"),
        "grasp_result": grasp_result,
        "cycle_time_s": current.get("cycle_time_s"),
        "placement_error_mm": current.get("placement_error_mm"),
        "overall_success_rate": overall_success_rate,
        "model_status": current.get("model_status"),
        "model_reason": current.get("model_reason"),
    }


def _dashboard_snapshot(run_dir: Path) -> dict[str, Any]:
    """Build the single, stable object consumed by the React dashboard."""
    status = _read_json_object(run_dir / "live_status.json")
    if status:
        normalized: dict[str, Any] = {
            "source": "simulation",
            "state": status.get("state", "running"),
            "phase": status.get("phase", "starting"),
            "mode": status.get("mode", "demo"),
            "updated_at": status.get("updated_at"),
            "current": status.get("current") if isinstance(status.get("current"), dict) else None,
            "progress": status.get("progress") if isinstance(status.get("progress"), dict) else {},
            "metrics": status.get("metrics") if isinstance(status.get("metrics"), dict) else {},
            "events": status.get("events") if isinstance(status.get("events"), list) else [],
            "planning": _planning_values(status.get("planning") or status.get("task_plan")),
            "result": status.get("state", "running"),
            "error": status.get("error"),
        }
    else:
        latest = _read_json_object(run_dir / "latest.json")
        archived = _archived_simulation_dashboard(run_dir)
        normalized = (
            _physical_dashboard(run_dir, latest)
            if latest
            else archived
            if archived is not None
            else {
                "source": "simulation",
                "state": "waiting",
                "phase": "starting",
                "mode": "demo",
                "updated_at": None,
                "current": None,
                "progress": {},
                "metrics": {},
                "events": [],
                "planning": _planning_values(None),
                "result": "pending",
                "error": None,
            }
        )

    current = normalized["current"] or {}
    metrics = normalized["metrics"]
    progress = normalized["progress"]
    completed = int(progress.get("completed_grabs") or 0)
    verified = int(metrics.get("verified_placements") or 0)
    success_rate = verified / completed if completed else None
    display = _display_values(current, str(normalized["phase"]), success_rate)
    timeline: list[dict[str, Any]] = []
    elapsed = 0.0
    raw_events = normalized["events"]
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        duration = event.get("cycle_time_s")
        duration_s = (
            float(duration)
            if isinstance(duration, (int, float)) and duration >= 0
            else 0.0
        )
        event_display = _display_values(event, str(event.get("stage") or normalized["phase"]), None)
        timeline.append(
            {
                "start_s": round(elapsed, 3),
                "end_s": round(elapsed + duration_s, 3),
                "stage": event.get("stage"),
                "grab": event.get("grab"),
                **event_display,
            }
        )
        elapsed += duration_s
    manifest = _artifact_manifest(run_dir)
    video = _video_file(run_dir)
    return {
        "schema_version": "1.0",
        "run": {
            "id": run_dir.name,
            "source": normalized["source"],
            "state": normalized["state"],
            "phase": normalized["phase"],
            "mode": normalized["mode"],
            "updated_at": normalized["updated_at"],
            "result": normalized["result"],
            "error": normalized["error"],
        },
        "display": display,
        "planning": normalized["planning"],
        "current": normalized["current"],
        "progress": progress,
        "metrics": metrics,
        "timeline": timeline,
        "timeline_basis": "cumulative_cycle_time",
        "media": {
            "video_url": "/api/video" if video is not None else None,
            "video_content_type": (
                mimetypes.guess_type(video.name)[0] or "application/octet-stream"
                if video is not None
                else None
            ),
            "latest_frames": manifest["latest_frames"],
        },
        "links": {
            "status": "/api/status",
            "events": "/api/events?stage=all",
            "metrics": "/api/metrics",
            "artifacts": "/api/artifacts",
            "sessions": "/api/sessions",
        },
    }


@dataclass(frozen=True)
class DashboardAPIConfig:
    run_dir: Path
    allowed_origins: frozenset[str]

    @classmethod
    def create(
        cls, run_dir: str | Path, allowed_origins: Iterable[str] = ()
    ) -> "DashboardAPIConfig":
        root = Path(run_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"run directory does not exist: {root}")
        return cls(root, frozenset(origin.rstrip("/") for origin in allowed_origins))


def _safe_file(root: Path, relative: str, *, suffixes: set[str]) -> Path | None:
    """Resolve a URL path below ``root`` without permitting traversal."""
    decoded = unquote(relative).replace("\\", "/")
    pure = PurePosixPath(decoded)
    if not decoded or pure.is_absolute() or ".." in pure.parts:
        return None
    candidate = root.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if candidate.suffix.lower() not in suffixes or not candidate.is_file():
        return None
    return candidate


def _latest_frame(run_dir: Path, stage: str, camera: str) -> Path | None:
    stage_dir = {"sort": "01_sort", "stack": "02_stack"}.get(stage)
    camera_dir = {"main": "frames", "side": "side_frames"}.get(camera)
    if stage_dir is None or camera_dir is None:
        return None
    folder = run_dir / stage_dir / camera_dir
    if not folder.is_dir():
        return None
    frames = (path for path in folder.glob("*.png") if path.is_file())
    return max(frames, key=lambda path: (path.stat().st_mtime_ns, path.name), default=None)


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SERVABLE_SUFFIXES:
            continue
        if path.name not in KEY_ARTIFACT_NAMES and path.suffix.lower() not in {
            ".mp4",
            ".webm",
        }:
            continue
        relative = path.relative_to(run_dir).as_posix()
        artifacts.append(
            {
                "path": relative,
                "url": "/artifacts/" + relative,
                "content_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "size_bytes": path.stat().st_size,
            }
        )
    artifacts.sort(key=lambda item: item["path"])
    latest_frames: dict[str, str] = {}
    for stage in ("sort", "stack"):
        for camera in ("main", "side"):
            if _latest_frame(run_dir, stage, camera) is not None:
                latest_frames[f"{stage}_{camera}"] = (
                    f"/api/latest-frame?stage={stage}&camera={camera}"
                )
    return {"artifacts": artifacts, "latest_frames": latest_frames}


def _session_manifest(run_dir: Path, *, limit: int = 40) -> dict[str, Any]:
    """List nearby run/session summaries without exposing filesystem paths."""
    sessions: list[dict[str, Any]] = []
    candidates = [path for path in run_dir.parent.iterdir() if path.is_dir()]
    if run_dir not in candidates:
        candidates.append(run_dir)
    for candidate in candidates:
        markers = (
            candidate / "live_status.json",
            candidate / "latest.json",
            candidate / "summary.json",
            candidate / "end_to_end_metrics.json",
        )
        if not any(path.is_file() for path in markers):
            continue
        snapshot = _dashboard_snapshot(candidate)
        updated_at = snapshot["run"].get("updated_at")
        mtimes = [path.stat().st_mtime for path in markers if path.is_file()]
        sessions.append(
            {
                "id": snapshot["run"]["id"],
                "source": snapshot["run"]["source"],
                "state": snapshot["run"]["state"],
                "phase": snapshot["run"]["phase"],
                "mode": snapshot["run"]["mode"],
                "result": snapshot["run"]["result"],
                "updated_at": updated_at,
                "sort_key": updated_at or max(mtimes, default=0),
                "steps_recorded": snapshot["metrics"].get("steps_recorded"),
                "last_latency_ms": snapshot["metrics"].get("last_latency_ms"),
                "last_action_safe": snapshot["metrics"].get("last_action_safe"),
                "completed_grabs": snapshot["progress"].get("completed_grabs"),
                "verified_placements": snapshot["metrics"].get("verified_placements"),
                "has_video": snapshot["media"]["video_url"] is not None,
                "is_current": candidate.resolve() == run_dir.resolve(),
            }
        )
    sessions.sort(key=lambda item: str(item.pop("sort_key", "")), reverse=True)
    return {"sessions": sessions[:limit], "current_run_id": run_dir.name}


def make_handler(config: DashboardAPIConfig) -> type[BaseHTTPRequestHandler]:
    class DashboardAPIHandler(BaseHTTPRequestHandler):
        server_version = "StackingDashboardAPI/1.0"

        def log_message(self, format: str, *args: object) -> None:
            # Keep the standard concise request log, but avoid DNS lookups.
            self.log_date_time_string()
            print(f"[dashboard-api] {self.client_address[0]} {format % args}")

        def _cors(self) -> None:
            origin = self.headers.get("Origin", "").rstrip("/")
            if origin and origin in config.allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                # Brev protects exposed HTTP ports with a Pomerium session
                # cookie.  Credentialed browser requests are permitted only
                # for an exact configured origin; wildcard origins are never
                # emitted by this server.
                self.send_header("Access-Control-Allow-Credentials", "true")
                self.send_header("Vary", "Origin")

        def _start(
            self,
            status: HTTPStatus,
            content_type: str,
            length: int,
            *,
            cache_control: str = "no-store",
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", cache_control)
            self.send_header("X-Content-Type-Options", "nosniff")
            self._cors()
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()

        def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
            self._start(status, "application/json; charset=utf-8", len(body))
            if self.command != "HEAD":
                self._write(body)

        def _write(self, body: bytes) -> bool:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return False
            return True

        def _error_json(self, status: HTTPStatus, message: str) -> None:
            self._json({"error": message, "status": int(status)}, status)

        def _serve_json_file(self, path: Path) -> None:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._error_json(HTTPStatus.NOT_FOUND, "run output is not available yet")
                return
            except (OSError, json.JSONDecodeError):
                self._error_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "run output is being updated; retry shortly",
                )
                return
            self._json(value)

        def _serve_file(self, path: Path) -> None:
            size = path.stat().st_size
            start, end = 0, max(0, size - 1)
            status = HTTPStatus.OK
            range_header = self.headers.get("Range")
            if range_header:
                match = _RANGE_RE.fullmatch(range_header.strip())
                if not match or size == 0:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                first, last = match.groups()
                if not first:
                    suffix_length = int(last or 0)
                    start = max(0, size - suffix_length)
                else:
                    start = int(first)
                    if last:
                        end = min(end, int(last))
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            length = 0 if size == 0 else end - start + 1
            headers = {"Accept-Ranges": "bytes"}
            if status == HTTPStatus.PARTIAL_CONTENT:
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._start(
                status,
                content_type,
                length,
                cache_control="private, max-age=1",
                extra_headers=headers,
            )
            if self.command == "HEAD" or length == 0:
                return
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    chunk = stream.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    if not self._write(chunk):
                        break
                    remaining -= len(chunk)

        def _route(self) -> None:
            parsed = urlsplit(self.path)
            endpoint = parsed.path.rstrip("/") or "/"
            if endpoint == "/health":
                self._json(
                    {
                        "status": "ok",
                        "run_id": config.run_dir.name,
                        "live_status_available": (config.run_dir / "live_status.json").is_file(),
                    }
                )
                return
            if endpoint in JSON_ENDPOINTS:
                self._serve_json_file(config.run_dir / JSON_ENDPOINTS[endpoint])
                return
            if endpoint == "/api/dashboard":
                self._json(_dashboard_snapshot(config.run_dir))
                return
            if endpoint == "/api/sessions":
                self._json(_session_manifest(config.run_dir))
                return
            if endpoint == "/api/feedback":
                self._json({"feedback": _feedback_rows(config.run_dir)})
                return
            if endpoint == "/api/video":
                video = _video_file(config.run_dir)
                if video is None:
                    self._error_json(HTTPStatus.NOT_FOUND, "recorded run video is not available")
                    return
                self._serve_file(video)
                return
            if endpoint == "/api/events":
                query = parse_qs(parsed.query)
                stage = query.get("stage", ["all"])[0]
                stage_dirs = {
                    "sort": ["01_sort"],
                    "stack": ["02_stack"],
                    "all": ["01_sort", "02_stack"],
                }.get(stage)
                if stage_dirs is None:
                    self._error_json(HTTPStatus.BAD_REQUEST, "stage must be sort, stack, or all")
                    return
                events: list[dict[str, Any]] = []
                for stage_dir in stage_dirs:
                    path = config.run_dir / stage_dir / "execution_events.json"
                    try:
                        rows = json.loads(path.read_text(encoding="utf-8"))
                    except FileNotFoundError:
                        continue
                    except (OSError, json.JSONDecodeError):
                        self._error_json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "event output is being updated; retry shortly",
                        )
                        return
                    label = "sort" if stage_dir == "01_sort" else "stack"
                    events.extend({"stage": label, **row} for row in rows)
                self._json({"events": events})
                return
            if endpoint == "/api/artifacts":
                self._json(_artifact_manifest(config.run_dir))
                return
            if endpoint == "/api/latest-frame":
                query = parse_qs(parsed.query)
                stage = query.get("stage", [""])[0]
                camera = query.get("camera", ["main"])[0]
                frame = _latest_frame(config.run_dir, stage, camera)
                if frame is None:
                    self._error_json(
                        HTTPStatus.NOT_FOUND,
                        "latest frame is not available for that stage and camera",
                    )
                    return
                self._serve_file(frame)
                return
            if endpoint.startswith("/artifacts/"):
                artifact = _safe_file(
                    config.run_dir,
                    endpoint.removeprefix("/artifacts/"),
                    suffixes=SERVABLE_SUFFIXES,
                )
                if artifact is None:
                    self._error_json(HTTPStatus.NOT_FOUND, "artifact was not found")
                    return
                self._serve_file(artifact)
                return
            if endpoint.startswith("/media/"):
                media = _safe_file(
                    config.run_dir,
                    endpoint.removeprefix("/media/"),
                    suffixes=MEDIA_SUFFIXES,
                )
                if media is None:
                    self._error_json(HTTPStatus.NOT_FOUND, "media was not found")
                    return
                self._serve_file(media)
                return
            self._error_json(HTTPStatus.NOT_FOUND, "endpoint was not found")

        def _read_request_json(self) -> Any:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if length <= 0 or length > _MAX_FEEDBACK_BYTES:
                raise ValueError("feedback request size is invalid")
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("request body must be valid JSON") from error

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._route()

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._route()

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            endpoint = (urlsplit(self.path).path.rstrip("/") or "/")
            if endpoint not in {"/api/feedback", "/api/plan"}:
                self._error_json(HTTPStatus.NOT_FOUND, "endpoint was not found")
                return
            origin = self.headers.get("Origin", "").rstrip("/")
            if config.allowed_origins and origin not in config.allowed_origins:
                self._error_json(HTTPStatus.FORBIDDEN, "origin is not allowed")
                return
            try:
                request_value = self._read_request_json()
                if endpoint == "/api/plan":
                    plan = _instruction_plan(request_value, run_dir=config.run_dir)
                    self._json(plan)
                    return
                feedback = _validated_feedback(request_value, run_id=config.run_dir.name)
                _append_feedback(config.run_dir, feedback)
            except ValueError as error:
                self._error_json(HTTPStatus.BAD_REQUEST, str(error))
                return
            except OSError:
                self._error_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "feedback storage is unavailable",
                )
                return
            self._json({"status": "recorded", "feedback": feedback}, HTTPStatus.CREATED)

        def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            origin = self.headers.get("Origin", "").rstrip("/")
            if origin not in config.allowed_origins:
                self._error_json(HTTPStatus.FORBIDDEN, "origin is not allowed")
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

    return DashboardAPIHandler


def create_server(
    run_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    allowed_origins: Iterable[str] = (),
) -> ThreadingHTTPServer:
    config = DashboardAPIConfig.create(run_dir, allowed_origins)
    server = ThreadingHTTPServer((host, port), make_handler(config))
    server.daemon_threads = True
    return server
