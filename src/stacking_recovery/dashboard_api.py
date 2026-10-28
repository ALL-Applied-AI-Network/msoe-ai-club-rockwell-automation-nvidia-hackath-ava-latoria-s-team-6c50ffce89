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
