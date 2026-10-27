import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sim" / "run_sort_then_stack.py"
SPEC = importlib.util.spec_from_file_location("run_sort_then_stack", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestSortThenStack(unittest.TestCase):
    def test_build_stack_stage_uses_sorted_positions(self) -> None:
        colours = ["yellow", "yellow", "blue", "blue", "red", "red"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sort_dir, stack_dir = root / "sort", root / "stack"
            sort_dir.mkdir()
            picks = []
            events = []
            after = {}
            for index, colour in enumerate(colours):
                name = f"SortCube{index}"
                picks.append({"detection_id": index, "category": "cube", "colour": colour})
                events.append({"detection_id": index, "tracked_object": name})
                after[name] = {
                    "position": [0.1 + index * 0.01, 0.2, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "centre": [0.1 + index * 0.01, 0.2, 0.05],
                    "yaw_deg": float(index),
                }
            (sort_dir / "sort_plan.json").write_text(json.dumps({"picks": picks}))
            (sort_dir / "execution_report.json").write_text(
                json.dumps({"task_complete": True, "picks": events, "after": after})
            )
            plan = MODULE.build_stack_stage(sort_dir, stack_dir, use_rl=True)
            state = json.loads((stack_dir / "scene_state.json").read_text())

        self.assertEqual([p["colour"] for p in plan["picks"]], colours)
        self.assertEqual(plan["picks"][0]["xy"], [0.1, 0.2])
        self.assertEqual(plan["closed_loop_policy"], "rl")
        self.assertEqual(len(state["objects"]), 6)

    def test_summary_scores_each_visual_prediction_against_physics(self) -> None:
        good = {
            "detection_id": 1,
            "tracked_object": "SortCube1",
            "pickup_verified": True,
            "placement_verified": True,
            "rl_visual_verification": {
                "predictions": [{"class": "GOOD", "confidence": 0.9}]
            },
        }
        missed = {
            "detection_id": 2,
            "tracked_object": "SortCube2",
            "pickup_verified": True,
            "placement_verified": False,
            "rl_visual_verification": {
                "predictions": [{"class": "SHIFTED", "confidence": 0.8}]
            },
        }
        summary = MODULE.summarize(
            {"task_complete": True, "verified_pickups": 1, "verified_placements": 1, "picks": [good]},
            {"task_complete": False, "verified_pickups": 1, "verified_placements": 0, "picks": [missed]},
        )
        self.assertFalse(summary["full_process_complete"])
        self.assertEqual(summary["classifier"]["accuracy"], 0.5)
        self.assertEqual(summary["classifier"]["confusion"]["true_positive"], 1)
        self.assertEqual(summary["classifier"]["confusion"]["false_negative"], 1)
        self.assertFalse(summary["tower_safety"]["all_existing_layers_stable"])

    def test_visual_benchmark_does_not_enable_rl_placement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / "source", root / "destination"
            source.mkdir()
            (source / "scene_state.json").write_text(json.dumps({"objects": {}}))
            (source / "sort_plan.json").write_text(
                json.dumps({"source": "Cosmos3-Nano", "picks": []})
            )
            MODULE.prepare_sort_stage(source, destination, use_rl=False)
            plan = json.loads((destination / "sort_plan.json").read_text())

        self.assertNotIn("closed_loop_policy", plan)

    def test_fast_stage_keeps_visual_classifier_and_adds_speedup(self) -> None:
        argv = MODULE.stage_argv(
            Path("run"),
            Path("frames"),
            physics_only=False,
            fast=True,
            checkpoint=Path("rl.json"),
            vision_config=Path("vision.yaml"),
        )

        self.assertEqual(argv[argv.index("--sim-speedup") + 1], "1.5")
        self.assertIn("--rl-vision-config", argv)
        self.assertIn("--rl-vision-observe-only", argv)
        self.assertNotIn("--physics-only", argv)

    def test_integrated_stage_gates_with_classifier_and_keeps_rendering(self) -> None:
        argv = MODULE.stage_argv(
            Path("run"),
            Path("frames"),
            physics_only=False,
            fast=False,
            checkpoint=Path("rl.json"),
            vision_config=Path("vision.yaml"),
            integrated=True,
            placement_config=Path("placement.json"),
        )

        self.assertEqual(argv[argv.index("--verification") + 1], "rules")
        self.assertIn("--rl-vision-config", argv)
        self.assertEqual(
            argv[argv.index("--placement-config") + 1], "placement.json"
        )
        self.assertNotIn("--rl-vision-observe-only", argv)
        self.assertNotIn("--physics-only", argv)

    def test_live_status_is_frontend_friendly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sort_dir = out / "01_sort"
            sort_dir.mkdir()
            event = {
                "detection_id": 0,
                "tracked_object": "SortCube0",
                "pickup_verified": True,
                "placement_verified": True,
                "closed_loop_grasp_attempts": 1,
                "rl_visual_verification": {
                    "quality": {"valid": True, "reason": None},
                    "predictions": [{"class": "GOOD", "confidence": 0.95}],
                    "gate": {"action": "proceed"},
                },
                "rl_placement_decision": {
                    "actions": ["CORRECT_LEFT", "PLACE_CENTER"],
                    "safe_stop": False,
                },
            }
            (sort_dir / "execution_events.json").write_text(json.dumps([event]))
            live = MODULE.LiveStatus(out, mode="fast")
            live.set_phase("sort")
            payload = json.loads((out / "live_status.json").read_text())

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["progress"]["completed_grabs"], 1)
        self.assertEqual(payload["current"]["predicted_class"], "GOOD")
        self.assertEqual(payload["current"]["rl"]["grasp_retries"], 1)
        self.assertEqual(payload["current"]["rl"]["action"], "PLACE_CENTER")
        self.assertEqual(
            payload["current"]["rl"]["actions"],
            ["CORRECT_LEFT", "PLACE_CENTER"],
        )
        self.assertEqual(payload["metrics"]["classifier_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
