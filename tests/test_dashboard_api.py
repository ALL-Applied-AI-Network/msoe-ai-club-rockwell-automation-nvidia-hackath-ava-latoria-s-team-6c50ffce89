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

from stacking_recovery.dashboard_api import (  # noqa: E402
    _dashboard_snapshot,
    _session_manifest,
    create_server,
)


class TestDashboardAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / "live_status.json").write_text(
            json.dumps(
                {
                    "state": "running",
                    "phase": "stack",
                    "mode": "integrated",
                    "planning": {
                        "planner": "deterministic",
                        "instruction": "Place the red cube on the blue cube.",
                        "action": "stack",
                        "source_color": "red",
                        "destination_color": "blue",
                        "shape": "cube",
                        "size_mm": 10,
                        "requires_clarification": False,
                    },
                    "progress": {"percent": 50, "completed_grabs": 2},
                    "current": {
                        "predicted_class": "GOOD",
                        "confidence": 0.94,
                        "pickup_verified": True,
                        "placement_verified": True,
                        "placement_error_mm": 3.2,
                        "cycle_time_s": 8.5,
                        "rl": {"action": "PLACE_CENTER", "actions": ["PLACE_CENTER"]},
                    },
                    "metrics": {"verified_placements": 2, "classifier_accuracy": 1.0},
                    "events": [
                        {
                            "stage": "stack",
                            "grab": 1,
                            "predicted_class": "GOOD",
                            "confidence": 0.94,
                            "pickup_verified": True,
                            "placement_verified": True,
                            "placement_error_mm": 3.2,
                            "cycle_time_s": 8.5,
                            "rl": {"action": "PLACE_CENTER", "actions": ["PLACE_CENTER"]},
                        }
                    ],
                }
            ),
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

    def post(self, path: str, payload: dict, *, origin: str = "http://localhost:5173"):
        body = json.dumps(payload).encode("utf-8")
        return urlopen(
            Request(
                self.base + path,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                    "Origin": origin,
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
        with self.get("/api/video", headers={"Range": "bytes=4-7"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"4567")

    def test_frontend_dashboard_contract(self) -> None:
        with self.get("/api/dashboard") as response:
            payload = json.load(response)

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["run"]["source"], "simulation")
        self.assertEqual(payload["display"]["phase"], "stack")
        self.assertEqual(payload["display"]["prediction"], "GOOD")
        self.assertEqual(payload["display"]["confidence"], 0.94)
        self.assertEqual(payload["display"]["rl_correction"], "PLACE_CENTER")
        self.assertEqual(payload["display"]["grasp_result"], "verified")
        self.assertEqual(payload["display"]["overall_success_rate"], 1.0)
        self.assertEqual(payload["planning"]["status"], "ready")
        self.assertEqual(payload["planning"]["planner"], "deterministic")
        self.assertEqual(payload["media"]["video_url"], "/api/video")
        self.assertEqual(payload["timeline"][0]["start_s"], 0.0)
        self.assertEqual(payload["timeline"][0]["end_s"], 8.5)
        self.assertEqual(payload["timeline"][0]["state_machine_step"], "verify placement")

    def test_physical_telemetry_uses_same_frontend_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "latest.json").write_text(
                json.dumps(
                    {
                        "wall_time_utc": "2026-10-28T18:00:00Z",
                        "rollout_index": 0,
                        "policy": {
                            "phase_after": "carry",
                            "result": "pending",
                            "policy_name": "SafeSixCubeStackV1",
                            "task_plan": {
                                "planner": "deterministic",
                                "instruction": "Stack all six cubes.",
                                "execution_skill": "SafeSixCubeStackV1",
                                "tower_order": ["yellow", "yellow", "blue", "blue", "red", "red"],
                                "requires_clarification": False,
                            },
                            "detections": {"red": {"x": 420, "y": -100}},
                        },
                        "cameras": {"table": {"shape": [720, 1280, 3]}},
                        "latency_ms": 12.5,
                        "action": {"row_count": 3, "all_targets_safe": True},
                    }
                ),
                encoding="utf-8",
            )
            (run / "summary.json").write_text(
                json.dumps({"steps_recorded": 12, "last_action_safe": True}),
                encoding="utf-8",
            )
            payload = _dashboard_snapshot(run)

        self.assertEqual(payload["run"]["source"], "physical_robot")
        self.assertEqual(payload["display"]["state_machine_step"], "carry")
        self.assertEqual(payload["display"]["selected_action"], "TCP motion chunk (3 targets)")
        self.assertEqual(payload["display"]["grasp_result"], "pending")
        self.assertEqual(payload["display"]["model_status"], "not_evaluated")
        self.assertEqual(payload["planning"]["status"], "ready")
        self.assertEqual(payload["planning"]["execution_skill"], "SafeSixCubeStackV1")
        self.assertEqual(payload["current"]["detection_count"], 1)
        self.assertEqual(payload["current"]["camera_names"], ["table"])
        self.assertTrue(payload["metrics"]["last_action_safe"])

    def test_completed_archived_run_uses_recorded_grasp_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "end_to_end_metrics.json").write_text(
                json.dumps(
                    {
                        "full_process_complete": True,
                        "classifier": {"scored_grabs": 1, "correct": 1, "accuracy": 1.0},
                        "grabs": [
                            {
                                "stage": "sort",
                                "grab": 1,
                                "predicted_class": "GOOD",
                                "confidence": 0.91,
                                "pickup_verified": True,
                                "placement_verified": True,
                                "placement_error_mm": 0.2,
                                "cycle_time_s": 7.0,
                                "visual_gate": "proceed",
                            },
                            {
                                "stage": "stack",
                                "grab": 1,
                                "predicted_class": None,
                                "confidence": None,
                                "pickup_verified": True,
                                "placement_verified": True,
                                "placement_error_mm": 0.4,
                                "cycle_time_s": 8.0,
                                "visual_gate": "safe_stop",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run / "original.webm").write_bytes(b"video")
            payload = _dashboard_snapshot(run)

        self.assertEqual(payload["run"]["state"], "completed")
        self.assertEqual(payload["run"]["mode"], "recorded")
        self.assertEqual(payload["progress"]["completed_grabs"], 2)
        self.assertEqual(payload["metrics"]["classifier_coverage"], 0.5)
        self.assertEqual(payload["timeline"][0]["start_s"], 0.0)
        self.assertEqual(payload["timeline"][1]["start_s"], 7.0)
        self.assertEqual(payload["timeline"][1]["end_s"], 15.0)
        self.assertEqual(payload["media"]["video_url"], "/api/video")

    def test_session_manifest_lists_physical_runs_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "physical-001"
            second = root / "physical-002"
            first.mkdir()
            second.mkdir()
            for index, run in enumerate((first, second), start=1):
                (run / "latest.json").write_text(
                    json.dumps(
                        {
                            "wall_time_utc": f"2026-10-30T17:0{index}:00Z",
                            "policy": {"phase_after": "carry", "result": "pending"},
                            "action": {"row_count": 2, "all_targets_safe": True},
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "summary.json").write_text(
                    json.dumps({"steps_recorded": index * 10, "last_action_safe": True}),
                    encoding="utf-8",
                )
            payload = _session_manifest(second)

        self.assertEqual(len(payload["sessions"]), 2)
        self.assertEqual(payload["sessions"][0]["id"], "physical-002")
        self.assertTrue(payload["sessions"][0]["is_current"])
        self.assertNotIn("path", payload["sessions"][0])

    def test_human_feedback_is_bounded_and_evidence_only(self) -> None:
        with self.post(
            "/api/feedback",
            {
                "run_id": self.run.name,
                "event_id": "stack-1-8.5",
                "kind": "stack",
                "label": "GOOD",
                "stage": "stack",
                "grab": 1,
                "model_prediction": "GOOD",
                "model_confidence": 0.94,
            },
        ) as response:
            payload = json.load(response)
        self.assertEqual(response.status, 201)
        self.assertEqual(payload["feedback"]["training_label"], "GOOD")
        self.assertTrue((self.run / "dashboard_feedback.jsonl").is_file())
        with self.get("/api/feedback") as response:
            rows = json.load(response)["feedback"]
        self.assertEqual(rows[0]["event_id"], "stack-1-8.5")

        with self.assertRaises(HTTPError) as ctx:
            self.post(
                "/api/feedback",
                {"event_id": "bad", "kind": "motion", "label": "GOOD"},
            )
        self.assertEqual(ctx.exception.code, 400)

    def test_instruction_studio_returns_dry_run_allowlisted_plan(self) -> None:
        with self.post(
            "/api/plan",
            {"instruction": "Place the red 10 mm cube on top of the blue 10 mm cube."},
        ) as response:
            payload = json.load(response)
        self.assertEqual(payload["mode"], "dry_run")
        self.assertEqual(payload["planning"]["status"], "ready")
        self.assertEqual(payload["planning"]["planner"], "deterministic")
        self.assertEqual(payload["plan"]["action"], "stack")
        self.assertIn("Plan preview only", payload["safety_boundary"])

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
