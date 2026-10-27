import json
import tempfile
import unittest
from pathlib import Path

from stacking_recovery.hardware import (
    DisabledHardwareAdapter,
    HardwareInterlockError,
    PoseTarget,
    ShadowRobotAdapter,
    load_hardware_adapter,
)


class TestHardwareAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = PoseTarget((0.3, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))

    def test_shadow_records_but_never_enables_motion(self) -> None:
        adapter = ShadowRobotAdapter()
        adapter.move_to_pose(self.pose, speed_scale=0.1)
        adapter.open_gripper()
        status = adapter.get_status()
        self.assertFalse(status.connected)
        self.assertFalse(status.motion_enabled)
        self.assertEqual([c.command for c in adapter.commands], ["move_to_pose", "open_gripper"])

    def test_shadow_rejects_speed_and_commands_after_stop(self) -> None:
        adapter = ShadowRobotAdapter()
        with self.assertRaises(HardwareInterlockError):
            adapter.move_to_pose(self.pose, speed_scale=0.5)
        adapter.stop()
        with self.assertRaises(HardwareInterlockError):
            adapter.close_gripper()

    def test_disabled_adapter_fails_closed(self) -> None:
        adapter = DisabledHardwareAdapter()
        with self.assertRaises(HardwareInterlockError):
            adapter.open_gripper()

    def test_loader_rejects_unreviewed_real_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hardware.json"
            path.write_text(json.dumps({"mode": "fanuc", "motion_enabled": True}))
            with self.assertRaises(HardwareInterlockError):
                load_hardware_adapter(path)

    def test_loader_allows_shadow_only_without_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hardware.json"
            path.write_text(json.dumps({"mode": "shadow", "motion_enabled": False}))
            self.assertIsInstance(load_hardware_adapter(path), ShadowRobotAdapter)


if __name__ == "__main__":
    unittest.main()
