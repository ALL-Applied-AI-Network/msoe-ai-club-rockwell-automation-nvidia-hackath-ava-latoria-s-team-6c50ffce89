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


if __name__ == "__main__":
    unittest.main()
