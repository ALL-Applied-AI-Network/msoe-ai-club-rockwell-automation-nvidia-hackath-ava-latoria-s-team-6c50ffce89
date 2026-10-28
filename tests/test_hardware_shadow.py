import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stacking_recovery.hardware import ShadowRobotAdapter, execute_shadow_plan


ROOT = Path(__file__).resolve().parents[1]


def example_plan() -> dict[str, object]:
    return {
        "picks": [
            {
                "eligible": True,
                "xy": [0.25, -0.05],
                "pick_z": 0.02,
                "place_xy": [0.40, 0.10],
                "place_z": 0.03,
            }
        ]
    }


class TestHardwareShadow(unittest.TestCase):
    def test_records_complete_no_motion_sequence(self) -> None:
        transcript = execute_shadow_plan(example_plan(), ShadowRobotAdapter())
        self.assertFalse(transcript["motion_sent"])
        self.assertEqual(
            [row["phase"] for row in transcript["commands"]],
            [
                "approach_pick",
                "open_before_pick",
                "descend_pick",
                "close_on_object",
                "lift",
                "approach_place",
                "descend_place",
                "release",
                "retreat",
            ],
        )

    def test_rejects_nonfinite_plan_coordinates(self) -> None:
        plan = example_plan()
        plan["picks"][0]["xy"][0] = float("nan")  # type: ignore[index]
        with self.assertRaises(ValueError):
            execute_shadow_plan(plan, ShadowRobotAdapter())

    def test_cli_writes_auditable_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = tmp_path / "plan.json"
            output = tmp_path / "shadow.json"
            plan.write_text(json.dumps(example_plan()), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_hardware_shadow.py"),
                    "--plan",
                    str(plan),
                    "--out",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["mode"], "shadow")
            self.assertFalse(value["final_status"]["connected"])


if __name__ == "__main__":
    unittest.main()
