from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.dashboard_api import create_server  # noqa: E402


class TestDashboardAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / "live_status.json").write_text(
            json.dumps({"state": "running", "progress": {"percent": 50}}),
            encoding="utf-8",
        )
        sort = self.run / "01_sort"
        (sort / "frames").mkdir(parents=True)
        (sort / "execution_events.json").write_text(
            json.dumps([{"grab": 1, "pickup_verified": True}]), encoding="utf-8"
        )
        (sort / "frames" / "f00001.png").write_bytes(b"old")
        (sort / "frames" / "f00002.png").write_bytes(b"new-frame")
        (self.run / "demo.mp4").write_bytes(b"0123456789")
        self.server = create_server(
            self.run,
            port=0,
            allowed_origins=["http://localhost:5173"],
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def get(self, path: str, *, headers: dict[str, str] | None = None):
        return urlopen(Request(self.base + path, headers=headers or {}), timeout=2)

    def options(self, path: str, *, origin: str):
        return urlopen(
            Request(
                self.base + path,
                method="OPTIONS",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "Range",
                },
            ),
            timeout=2,
        )

    def test_status_health_events_and_manifest(self) -> None:
        with self.get("/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")
        with self.get("/api/status") as response:
            self.assertEqual(json.load(response)["state"], "running")
        with self.get("/api/events?stage=sort") as response:
            payload = json.load(response)
            self.assertEqual(payload["events"][0]["stage"], "sort")
        with self.get("/api/artifacts") as response:
            payload = json.load(response)
            paths = {item["path"] for item in payload["artifacts"]}
            self.assertIn("demo.mp4", paths)
            self.assertEqual(
                payload["latest_frames"]["sort_main"],
                "/api/latest-frame?stage=sort&camera=main",
            )

    def test_latest_frame_and_video_range(self) -> None:
        with self.get("/api/latest-frame?stage=sort&camera=main") as response:
            self.assertEqual(response.read(), b"new-frame")
        with self.get("/media/demo.mp4", headers={"Range": "bytes=2-5"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
            self.assertEqual(response.read(), b"2345")

    def test_cors_is_allowlisted(self) -> None:
        with self.get(
            "/api/status", headers={"Origin": "http://localhost:5173"}
        ) as response:
            self.assertEqual(
                response.headers["Access-Control-Allow-Origin"],
                "http://localhost:5173",
            )
            self.assertEqual(
                response.headers["Access-Control-Allow-Credentials"], "true"
            )
        with self.options(
            "/api/status", origin="http://localhost:5173"
        ) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(
                response.headers["Access-Control-Allow-Origin"],
                "http://localhost:5173",
            )
            self.assertEqual(
                response.headers["Access-Control-Allow-Credentials"], "true"
            )
            self.assertIn("GET", response.headers["Access-Control-Allow-Methods"])
        with self.get(
            "/api/status", headers={"Origin": "https://untrusted.example"}
        ) as response:
            self.assertIsNone(response.headers["Access-Control-Allow-Origin"])
            self.assertIsNone(response.headers["Access-Control-Allow-Credentials"])

    def test_options_rejects_untrusted_origin(self) -> None:
        with self.assertRaises(HTTPError) as ctx:
            self.options("/api/status", origin="https://untrusted.example")
        self.assertEqual(ctx.exception.code, 403)

    def test_path_traversal_and_unsupported_files_are_blocked(self) -> None:
        outside = self.run.parent / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        try:
            for path in (
                "/artifacts/%2e%2e/secret.txt",
                "/artifacts/01_sort/%2e%2e/%2e%2e/secret.txt",
            ):
                with self.assertRaises(HTTPError) as ctx:
                    self.get(path)
                self.assertEqual(ctx.exception.code, 404)
        finally:
            outside.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
