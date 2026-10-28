import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "data" / "fanuc_er4ia_hackathon_starter_v05 (1)" / "fanuc_er4ia_sim"
sys.path.insert(0, str(SIM_DIR))
SPEC = importlib.util.spec_from_file_location("stack_safety", SIM_DIR / "stack_safety.py")
assert SPEC and SPEC.loader
STACK_SAFETY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STACK_SAFETY)


def pose(x=0.28, y=0.10, z=0.93, yaw=0.0, tilt=0.0):
    return {"centre": [x, y, z], "yaw_deg": yaw, "tilt_deg": tilt}


class TestStackSafety(unittest.TestCase):
    def test_stack_path_slows_for_final_30_mm(self) -> None:
        pick = {
            "xy": [0.2, 0.0],
            "pick_z": 0.01,
            "place_xy": [0.28, 0.10],
            "place_z": 0.08,
            "category": "cube",
            "stack_level": 3,
            "spread_dir": [0.0, 1.0, 0.0],
            "place_spread_dir": [0.0, 1.0, 0.0],
        }
        segments = STACK_SAFETY.segments(pick, [0.3, 0.0, 0.24])
        names = [segment[0] for segment in segments]

        self.assertIn("prelower", names)
        self.assertIn("final_lower", names)
        self.assertNotIn("lower", names)
        final = segments[names.index("final_lower")]
        self.assertAlmostEqual(final[1][2] - final[2][2], 0.03)

    def test_existing_tower_shift_fails_closed(self) -> None:
        entries = [{"tracked_object": "SortCube0"}]
        before = {"SortCube0": pose()}
        after = {"SortCube0": pose(x=0.282)}

        result = STACK_SAFETY.measure_stack_disturbance(before, after, entries)

        self.assertFalse(result["tower_disturbance_verified"])
        self.assertAlmostEqual(result["max_existing_layer_shift_mm"], 2.0)

    def test_first_layer_has_no_existing_tower_to_disturb(self) -> None:
        result = STACK_SAFETY.measure_stack_disturbance({}, {}, [])

        self.assertTrue(result["tower_disturbance_verified"])
        self.assertEqual(result["existing_layers_checked"], 0)


if __name__ == "__main__":
    unittest.main()
