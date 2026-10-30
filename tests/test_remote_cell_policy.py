import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


POLICY_PATH = Path(__file__).resolve().parents[1] / "hardware" / "remote_cell" / "policy.py"
SPEC = importlib.util.spec_from_file_location("remote_cell_policy", POLICY_PATH)
assert SPEC and SPEC.loader
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class TestRemoteCellPolicy(unittest.TestCase):
    def test_known_instruction_uses_deterministic_plan_without_network(self):
        with mock.patch.object(
            POLICY.urlrequest,
            "urlopen",
            side_effect=AssertionError("network should not be used"),
        ):
            plan = POLICY.resolve_task_plan(
                "Place the red 10 mm cube on top of the blue 10 mm cube."
            )

        self.assertEqual(plan["planner"], "deterministic")
        self.assertEqual(plan["action"], "stack")
        self.assertEqual(plan["source_color"], "red")
        self.assertEqual(plan["destination_color"], "blue")

    def test_gemini_structured_plan_is_validated_without_exposing_key(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                task = {
                    "action": "stack",
                    "source_color": "blue",
                    "destination_color": "red",
                    "shape": "cube",
                    "size_mm": 10,
                    "requires_clarification": False,
                }
                body = {
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(task)}]}}
                    ]
                }
                return json.dumps(body).encode("utf-8")

        old_key = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ["GEMINI_API_KEY"] = "test-key-never-log"
            with mock.patch.object(
                POLICY.urlrequest,
                "urlopen",
                return_value=FakeResponse(),
            ) as call:
                plan = POLICY.resolve_task_plan(
                    "Stack the blue ten millimeter cube above the red cube."
                )

            self.assertEqual(plan["planner"], "gemini")
            self.assertEqual(plan["source_color"], "blue")
            request = call.call_args.args[0]
            self.assertNotIn("test-key-never-log", request.full_url)
        finally:
            if old_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_gemini_six_cube_plan_selects_only_allowlisted_skill(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                task = {
                    "action": "stack_all",
                    "shape": "cube",
                    "size_mm": 10,
                    "cube_count": 6,
                    "stack_colors_bottom_to_top": [
                        "yellow",
                        "yellow",
                        "blue",
                        "blue",
                        "red",
                        "red",
                    ],
                    "requires_clarification": False,
                }
                body = {
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(task)}]}}
                    ]
                }
                return json.dumps(body).encode("utf-8")

        old_key = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ["GEMINI_API_KEY"] = "test-key-never-log"
            with mock.patch.object(
                POLICY.urlrequest,
                "urlopen",
                return_value=FakeResponse(),
            ) as call:
                plan = POLICY.resolve_task_plan(
                    "Stack all six cubes into one tower.",
                    force_gemini=True,
                )

            self.assertEqual(plan["planner"], "gemini")
            self.assertEqual(plan["execution_skill"], "SafeSixCubeStackV1")
            self.assertEqual(plan["cube_count"], 6)
            self.assertEqual(
                tuple(plan["stack_colors_bottom_to_top"]), POLICY.SIX_CUBE_ORDER
            )
            payload = json.loads(call.call_args.args[0].data.decode("utf-8"))
            schema = payload["generationConfig"]["responseFormat"]["text"]["schema"]
            self.assertEqual(schema["properties"]["action"]["enum"], ["stack_all"])
        finally:
            if old_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_canonical_six_cube_instruction_uses_backend_skill_without_network(self):
        with mock.patch.object(
            POLICY.urlrequest,
            "urlopen",
            side_effect=AssertionError("network should not be used"),
        ):
            plan = POLICY.resolve_task_plan(POLICY.SIX_CUBE_INSTRUCTION)

        self.assertEqual(plan["planner"], "deterministic")
        self.assertEqual(plan["execution_skill"], "SafeSixCubeStackV1")
        self.assertEqual(plan["cube_count"], 6)
        self.assertEqual(
            tuple(plan["stack_colors_bottom_to_top"]), POLICY.SIX_CUBE_ORDER
        )
        self.assertEqual(plan["instruction"], POLICY.SIX_CUBE_INSTRUCTION)

    def test_gemini_six_cube_plan_rejects_unsafe_order(self):
        with self.assertRaises(POLICY.InstructionError):
            POLICY._validated_six_cube_task_plan(
                {
                    "action": "stack_all",
                    "shape": "cube",
                    "size_mm": 10,
                    "cube_count": 6,
                    "stack_colors_bottom_to_top": [
                        "red",
                        "red",
                        "blue",
                        "blue",
                        "yellow",
                        "yellow",
                    ],
                    "requires_clarification": False,
                },
                instruction="Stack all six cubes.",
                planner="gemini",
            )

    def test_ambiguous_or_unsupported_instruction_fails_closed(self):
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            policy = POLICY.Policy()
            policy._calibration_error = None
            obs = {
                "trial": {"instruction": "Move something somewhere."},
                "images": {
                    "table": np.zeros((360, 640, 3), dtype=np.uint8),
                    "wrist": np.zeros((360, 640, 3), dtype=np.uint8),
                },
                "tcp_pose": [300, 0, -250, -180, 0, 0],
                "joints": [0, 0, 0, 0, 0, 0, 0],
            }

            action = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)

            self.assertEqual(policy.phase, "stopped")
            self.assertIn("instruction rejected", policy.reason)
            self.assertTrue(all(POLICY.tcp_is_safe(row) for row in action))
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_color_centers_find_small_red_and_blue_regions(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[20:32, 30:44] = (230, 20, 20)
        image[70:84, 150:166] = (15, 30, 220)
        red = POLICY.color_center(image, "red")
        blue = POLICY.color_center(image, "blue")
        self.assertIsNotNone(red)
        self.assertIsNotNone(blue)
        self.assertAlmostEqual(red[0], 36.5, delta=1.0)
        self.assertAlmostEqual(blue[0], 157.5, delta=1.0)

    def test_color_components_ignore_border_clutter(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[0:20, 0:70] = (230, 20, 20)
        image[40:54, 50:66] = (230, 20, 20)
        image[75:91, 130:148] = (230, 20, 20)

        components = POLICY.color_components(image, "red")

        self.assertEqual(len(components), 2)
        centers = [component["center"] for component in components]
        self.assertTrue(any(abs(u - 57.5) < 1 and abs(v - 46.5) < 1 for u, v in centers))
        self.assertTrue(any(abs(u - 138.5) < 1 and abs(v - 82.5) < 1 for u, v in centers))

    def test_wrist_color_center_allows_close_cube(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[25:75, 65:125] = (230, 20, 20)

        self.assertEqual(POLICY.color_components(image, "red"), [])
        center = POLICY.color_center(image, "red")
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center[0], 94.5, delta=1.0)
        self.assertAlmostEqual(center[1], 49.5, delta=1.0)

    def test_workspace_rejects_out_of_box_and_out_of_reach(self):
        self.assertTrue(POLICY.tcp_is_safe([450, 0, -250, -180, 0, 0, 0]))
        self.assertFalse(POLICY.tcp_is_safe([149, 0, -250, -180, 0, 0, 0]))
        self.assertFalse(POLICY.tcp_is_safe([600, 0, -250, -180, 0, 0, 0]))

    def test_interpolation_is_bounded_and_finishes_at_target(self):
        rows = POLICY.interpolate_tcp(
            [400, 0, -250, -180, 0, 0],
            [430, 20, -280, -180, 0, 0, 1],
            step_mm=10,
        )
        self.assertGreater(len(rows), 1)
        self.assertLessEqual(len(rows), 100)
        self.assertEqual(rows[-1], [430.0, 20.0, -280.0, -180.0, 0.0, 0.0, 1.0])
        self.assertTrue(all(POLICY.tcp_is_safe(row) for row in rows))

    def test_operator_xy_parser_requires_two_finite_values(self):
        self.assertEqual(POLICY.parse_xy("420,-100"), (420.0, -100.0))
        self.assertIsNone(POLICY.parse_xy(None))
        with self.assertRaises(ValueError):
            POLICY.parse_xy("420")
        with self.assertRaises(ValueError):
            POLICY.parse_xy("nan,100")

    def test_operator_fallback_must_be_complete_and_safe(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ["REMOTE_ARM_RED_XY_MM"] = "420,-100"
            os.environ.pop("REMOTE_ARM_BLUE_XY_MM", None)
            with self.assertRaises(ValueError):
                POLICY.Policy()

            os.environ["REMOTE_ARM_BLUE_XY_MM"] = "500,100"
            policy = POLICY.Policy()
            self.assertEqual(policy._fixed_targets["red"], (420.0, -100.0))
            self.assertEqual(policy._fixed_targets["blue"], (500.0, 100.0))
            self.assertTrue(policy._operator_fallback_active)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_operator_fallback_replaces_only_an_unsafe_visible_detection(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ["REMOTE_ARM_RED_XY_MM"] = "420,-100"
            os.environ["REMOTE_ARM_BLUE_XY_MM"] = "500,100"
            policy = POLICY.Policy()
            policy._cams = {"table": {"size": (640, 360)}}
            policy._pixel_to_base = lambda *_: np.array([0.9, 0.9, 0.0])
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[100:112, 200:212] = (230, 20, 20)

            self.assertEqual(policy._table_xy(image, "red"), (420.0, -100.0))
            self.assertEqual(
                policy._last_detections["red"]["source"],
                "operator_confirmed_fallback",
            )
            self.assertIsNone(
                policy._table_xy(image, "red", allow_operator_fallback=False)
            )
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_table_detector_checks_all_components_and_uses_safe_one(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            for name in POLICY.FIXED_TARGET_ENV.values():
                os.environ.pop(name, None)
            policy = POLICY.Policy()
            policy._cams = {"table": {"size": (640, 360)}}

            def fake_pixel_to_base(_camera, uv):
                # The larger component is intentionally unsafe.  The smaller
                # one is the cube and maps to its expected physical location.
                if uv[0] > 400:
                    return np.array([0.9, 0.9, 0.0])
                return np.array([0.42, -0.10, 0.0])

            policy._pixel_to_base = fake_pixel_to_base
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[90:104, 180:194] = (230, 20, 20)
            image[250:280, 500:530] = (230, 20, 20)

            self.assertEqual(policy._table_xy(image, "red"), (420.0, -100.0))
            detection = policy._last_detections["red"]
            self.assertEqual(detection["component_count"], 2)
            self.assertEqual(detection["safe_component_count"], 1)
            self.assertEqual(detection["source"], "calibrated_camera")
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_operator_anchor_rejects_distant_in_bounds_false_positive(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ["REMOTE_ARM_RED_XY_MM"] = "420,-100"
            os.environ["REMOTE_ARM_BLUE_XY_MM"] = "500,100"
            policy = POLICY.Policy()
            policy._cams = {"table": {"size": (640, 360)}}
            policy._pixel_to_base = lambda *_: np.array([0.25, 0.25, 0.0])
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[100:114, 200:214] = (230, 20, 20)

            self.assertEqual(policy._table_xy(image, "red"), (420.0, -100.0))
            self.assertEqual(
                policy._last_detections["red"]["source"],
                "operator_confirmed_fallback",
            )
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_supervised_fallback_sequence_stays_safe_and_reports_unverified(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ["REMOTE_ARM_RED_XY_MM"] = "420,-100"
            os.environ["REMOTE_ARM_BLUE_XY_MM"] = "500,100"
            policy = POLICY.Policy()
            policy._calibration_error = None
            policy._cams = {"table": {"size": (640, 360)}}
            policy._pixel_to_base = lambda *_: np.array([0.9, 0.9, 0.0])

            table = np.zeros((360, 640, 3), dtype=np.uint8)
            table[100:112, 200:212] = (230, 20, 20)
            table[200:212, 400:412] = (20, 30, 230)
            wrist = np.zeros_like(table)
            wrist[170:186, 310:326] = (230, 20, 20)
            obs = {
                "images": {"table": table, "wrist": wrist},
                "tcp_pose": [300, 0, -90, -180, 0, 0],
                "joints": [0, 0, 0, 0, 0, 0, 0],
            }

            for _ in range(9):
                action = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
                self.assertTrue(all(POLICY.tcp_is_safe(row) for row in action))
                obs["tcp_pose"] = action[-1, :6].tolist()
                obs["joints"][-1] = action[-1, 6]

            self.assertEqual(policy.phase, "complete")
            self.assertEqual(policy.result, "completed_unverified")
            self.assertIn("fallback", policy.reason)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_calibrated_cube_disappearance_with_closed_gripper_verifies_pick(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            for name in POLICY.FIXED_TARGET_ENV.values():
                os.environ.pop(name, None)
            policy = POLICY.Policy()
            policy._calibration_error = None
            policy._cams = {"table": {"size": (640, 360)}}

            def fake_pixel_to_base(_camera, uv):
                if uv[0] < 300:
                    return np.array([0.42, -0.10, 0.0])
                return np.array([0.50, 0.10, 0.0])

            policy._pixel_to_base = fake_pixel_to_base
            table = np.zeros((360, 640, 3), dtype=np.uint8)
            table[100:112, 200:212] = (230, 20, 20)
            table[200:212, 400:412] = (20, 30, 230)
            obs = {
                "images": {"table": table, "wrist": np.zeros_like(table)},
                "tcp_pose": [300, 0, -90, -180, 0, 0],
                "joints": [0, 0, 0, 0, 0, 0, 0],
            }

            for _ in range(4):
                action = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
                obs["tcp_pose"] = action[-1, :6].tolist()
                obs["joints"][-1] = action[-1, 6]

            # The calibrated red cube was visible before pickup.  Once the
            # closed gripper lifts, its disappearance from the table is valid
            # verification even when the wrist camera cannot see a 10 mm cube.
            obs["images"]["table"] = np.zeros_like(table)
            obs["images"]["table"][200:212, 400:412] = (20, 30, 230)
            action = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)

            self.assertEqual(policy.phase, "descend_place")
            self.assertEqual(
                policy._pick_verification_source,
                "table_disappearance_with_closed_gripper",
            )
            self.assertTrue(all(POLICY.tcp_is_safe(row) for row in action))
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_repeated_chunk_uses_last_target_without_z_bounce(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            for name in POLICY.FIXED_TARGET_ENV.values():
                os.environ.pop(name, None)
            policy = POLICY.Policy()
            policy._calibration_error = None
            policy._cams = {"table": {"size": (640, 360)}}

            def fake_pixel_to_base(_camera, uv):
                return (
                    np.array([0.42, -0.10, 0.0])
                    if uv[0] < 300
                    else np.array([0.50, 0.10, 0.0])
                )

            policy._pixel_to_base = fake_pixel_to_base
            table = np.zeros((360, 640, 3), dtype=np.uint8)
            table[100:112, 200:212] = (230, 20, 20)
            table[200:212, 400:412] = (20, 30, 230)
            obs = {
                "images": {"table": table, "wrist": np.zeros_like(table)},
                "tcp_pose": [300, 0, -90, -180, 0, 0],
                "joints": [0, 0, 0, 0, 0, 0, 0],
            }

            first = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
            self.assertEqual(policy.phase, "descend_pick")
            # Simulate the physical observation lagging midway through the
            # first chunk.  The follow-up must repeat the prior endpoint, not
            # interpolate backward from this stale pose.
            obs["tcp_pose"] = [380, -60, -180, -180, 0, 0]
            follow_up = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)

            self.assertEqual(policy.phase, "descend_pick")
            self.assertEqual(len(follow_up), 1)
            np.testing.assert_allclose(follow_up[0], first[-1])
            self.assertEqual(follow_up[0, 2], POLICY.SAFE_Z_MM)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_six_cube_instruction_and_yellow_detector(self):
        self.assertTrue(POLICY.is_six_cube_instruction("Stack all six cubes into one tower."))
        self.assertTrue(POLICY.is_six_cube_instruction("Build a tower with 6 cubes."))
        self.assertFalse(
            POLICY.is_six_cube_instruction(
                "Place the red 10 mm cube on top of the blue 10 mm cube."
            )
        )

        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[30:44, 40:56] = (220, 195, 20)
        yellow = POLICY.color_center(image, "yellow")
        self.assertIsNotNone(yellow)
        self.assertAlmostEqual(yellow[0], 47.5, delta=1.0)
        self.assertAlmostEqual(yellow[1], 36.5, delta=1.0)

    @staticmethod
    def _six_cube_table():
        table = np.zeros((360, 640, 3), dtype=np.uint8)
        components = [
            ("yellow", 100, 80, (220, 195, 20)),
            ("yellow", 520, 80, (220, 195, 20)),
            ("blue", 180, 180, (20, 30, 230)),
            ("blue", 460, 180, (20, 30, 230)),
            ("red", 260, 280, (230, 20, 20)),
            ("red", 380, 280, (230, 20, 20)),
        ]
        for _color, u, v, rgb in components:
            table[v : v + 12, u : u + 12] = rgb
        return table

    def test_six_cube_plan_detects_duplicates_and_increases_layer_height(self):
        policy = POLICY.Policy()
        policy._calibration_error = None
        policy._cams = {"table": {"size": (640, 360)}}

        def fake_pixel_to_base(_camera, uv):
            return np.array(
                [
                    (250.0 + float(uv[0]) * 0.45) / 1000.0,
                    ((float(uv[1]) - 180.0) * 0.50) / 1000.0,
                    0.0,
                ]
            )

        policy._pixel_to_base = fake_pixel_to_base
        plan = policy._build_six_cube_plan(self._six_cube_table())

        self.assertEqual(len(plan), 5)
        self.assertEqual(
            [step["color"] for step in plan],
            ["yellow", "blue", "blue", "red", "red"],
        )
        self.assertEqual(
            [step["place_z_mm"] for step in plan],
            [-319.0, -309.0, -299.0, -289.0, -279.0],
        )
        self.assertTrue(
            all(
                POLICY.tcp_is_safe(
                    policy._target(
                        tuple(step["tower_xy_mm"]), step["place_z_mm"], 1.0
                    )
                )
                for step in plan
            )
        )

    def test_canonical_six_cube_instruction_stops_before_motion_when_layout_missing(self):
        old_flag = os.environ.get(POLICY.SIX_CUBE_ENV)
        try:
            os.environ[POLICY.SIX_CUBE_ENV] = "1"
            policy = POLICY.Policy()
            policy._calibration_error = None
            policy._cams = {"table": {"size": (640, 360)}}
            table = np.zeros((360, 640, 3), dtype=np.uint8)
            obs = {
                "trial": {"instruction": POLICY.SIX_CUBE_INSTRUCTION},
                "images": {"table": table, "wrist": np.zeros_like(table)},
                "tcp_pose": [300, 0, -250, -180, 0, 0],
                "joints": [0, 0, 0, 0, 0, 0, 0],
            }

            action = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)

            self.assertEqual(policy.phase, "stopped")
            self.assertIn("requires exactly 2 safe yellow cubes", policy.reason)
            self.assertTrue(all(POLICY.tcp_is_safe(row) for row in action))
            np.testing.assert_allclose(action[-1, :6], obs["tcp_pose"])
        finally:
            if old_flag is None:
                os.environ.pop(POLICY.SIX_CUBE_ENV, None)
            else:
                os.environ[POLICY.SIX_CUBE_ENV] = old_flag

    def test_six_cube_mode_completes_five_safe_verified_moves(self):
        old_flag = os.environ.get(POLICY.SIX_CUBE_ENV)
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ[POLICY.SIX_CUBE_ENV] = "1"
            for name in POLICY.FIXED_TARGET_ENV.values():
                os.environ.pop(name, None)
            policy = POLICY.Policy()
            policy._calibration_error = None
            tower = (420.0, 0.0)
            sources = [
                (300.0, -120.0),
                (340.0, -80.0),
                (380.0, -120.0),
                (460.0, -80.0),
                (500.0, -120.0),
            ]
            colors = ["yellow", "blue", "blue", "red", "red"]
            plan = [
                {
                    "layer": layer,
                    "color": color,
                    "source_xy_mm": source,
                    "tower_xy_mm": tower,
                    "place_z_mm": POLICY.PICK_Z_MM
                    + POLICY.CUBE_HEIGHT_MM * layer,
                }
                for layer, (color, source) in enumerate(
                    zip(colors, sources), start=1
                )
            ]

            def fake_build(_table):
                policy._six_tower_xy_mm = tower
                return plan

            policy._build_six_cube_plan = fake_build

            def fake_table_xys(_table, _color):
                if policy.phase == "verify_pick" or policy._six_current is None:
                    return [tower]
                return [tower, tuple(policy._six_current["source_xy_mm"])]

            policy._table_xys = fake_table_xys
            table = np.zeros((360, 640, 3), dtype=np.uint8)
            obs = {
                "trial": {"instruction": "Stack all six cubes into one tower."},
                "images": {"table": table, "wrist": np.zeros_like(table)},
                "tcp_pose": [300, 0, -250, -180, 0, 0],
                "joints": [0, 0, 0, 0, 0, 0, 0],
            }

            for _ in range(60):
                action = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
                self.assertTrue(all(POLICY.tcp_is_safe(row) for row in action))
                obs["tcp_pose"] = action[-1, :6].tolist()
                obs["joints"][-1] = action[-1, 6]
                if policy.phase == "complete":
                    break

            self.assertEqual(policy.phase, "complete")
            self.assertEqual(policy.result, "success")
            self.assertEqual(len(policy._six_cube_verified_layers), 5)
            self.assertEqual(
                [row["place_z_mm"] for row in policy._six_cube_verified_layers],
                [-319.0, -309.0, -299.0, -289.0, -279.0],
            )
        finally:
            if old_flag is None:
                os.environ.pop(POLICY.SIX_CUBE_ENV, None)
            else:
                os.environ[POLICY.SIX_CUBE_ENV] = old_flag
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
