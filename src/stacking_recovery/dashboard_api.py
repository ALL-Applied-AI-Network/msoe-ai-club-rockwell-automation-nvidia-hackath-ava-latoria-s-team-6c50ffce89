"""Dependency-free, read-only HTTP bridge for the demo dashboard.

The simulator owns the run directory and writes files atomically.  This module
only reads that directory; it never starts, stops, or controls the robot.
"""

from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass
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


def _read_json_object(path: Path) -> dict[str, Any]:
    """Return one JSON object, or an empty object while a producer is writing."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
        "result": result,
        "error": latest.get("error") or summary.get("telemetry_error"),
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
            "result": status.get("state", "running"),
            "error": status.get("error"),
        }
    else:
        latest = _read_json_object(run_dir / "latest.json")
        normalized = (
            _physical_dashboard(run_dir, latest)
            if latest
            else {
                "source": "simulation",
                "state": "waiting",
                "phase": "starting",
                "mode": "demo",
                "updated_at": None,
                "current": None,
                "progress": {},
                "metrics": {},
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
    raw_events = status.get("events") if isinstance(status.get("events"), list) else []
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
                self.wfile.write(body)

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
                    self.wfile.write(chunk)
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

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._route()

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._route()

        def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            origin = self.headers.get("Origin", "").rstrip("/")
            if origin not in config.allowed_origins:
                self._error_json(HTTPStatus.FORBIDDEN, "origin is not allowed")
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
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
