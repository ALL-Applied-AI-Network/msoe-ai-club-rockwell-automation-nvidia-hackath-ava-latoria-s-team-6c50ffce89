#!/usr/bin/env python3
"""Serve one simulator run to a read-only React/TypeScript dashboard."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_local_gemini_settings() -> None:
    settings = ROOT / ".env.local"
    if not settings.is_file():
        return
    allowed = {"GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_TIMEOUT_SECONDS"}
    for raw_line in settings.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in allowed:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_load_local_gemini_settings()

from stacking_recovery.dashboard_api import create_server  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="sort-then-stack output directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="allowed dashboard origin; repeat for multiple origins",
    )
    args = parser.parse_args()
    server = create_server(
        args.run,
        host=args.host,
        port=args.port,
        allowed_origins=args.allow_origin,
    )
    host, port = server.server_address[:2]
    print(f"Dashboard API: http://{host}:{port}", flush=True)
    print(f"Run directory: {args.run.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
