"""Execute the Cosmos sort plan with physical grasps and pose-based reporting.

Run ./run_sort_pipeline.sh RUN_DIR to capture, perceive, plan, render and encode.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import scene_config as C
from closed_loop_support import ensure_stacking_recovery_on_path, rgba_to_bgr
import fk
import ik
from cube_pose import square_yaw
from stack_safety import measure_stack_disturbance, segments


def ease(t):
    return t * t * (3 - 2 * t)


def turn(a, b, t):
    a, b = np.array(a, float), np.array(b, float)
    angle = np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a[:2], b[:2])) * t
    c, s = np.cos(angle), np.sin(angle)
    return np.array([a[0] * c - a[1] * s, a[0] * s + a[1] * c, 0.0])


def expected_placement_centre_z(category):
    """Return the resting object-centre height for a non-stacked placement."""
    half_height = {
        "cube": C.CUBE_EDGE / 2,
        "cap": C.CAP_HEIGHT / 2,
        "vial": C.VIAL_RADIUS,
    }[category]
    return C.TABLE_TOP_Z + C.SURFACE_DZ + half_height


def verify_cube_stack(positions, entries):
    """Check site, layer heights and actual adjacent-cube support geometry."""
    centres = []
    for entry in entries:
        name = entry.get("tracked_object")
        if name not in positions:
            return False
        pose = positions[name]
        if "yaw_deg" in pose and (
            abs(square_yaw(pose["yaw_deg"])) >= 8 or pose.get("tilt_deg", 0) >= 5
        ):
            return False
        centre = np.array(pose["centre"])
        if np.linalg.norm(centre[:2] - C.CUBE_STACK_XY) > C.CUBE_STACK_SITE_TOLERANCE:
            return False
        height = (
            C.TABLE_TOP_Z + C.SURFACE_DZ + (entry["stack_level"] + 0.5) * C.CUBE_EDGE
        )
        if abs(centre[2] - height) > 0.0015:
            return False
        if (
            centres
            and np.linalg.norm(centre[:2] - centres[-1][:2])
            > C.CUBE_STACK_LAYER_TOLERANCE
        ):
            return False
        centres.append(centre)
    return True


def validate_plan(plan, *, allow_collection_ground_truth=False):
    source_ok = plan.get("source") == "Cosmos3-Nano"
    collection_source_ok = (
        allow_collection_ground_truth
        and plan.get("source") == "Isaac-sim-ground-truth-data-collection-only"
        and plan.get("data_collection_only") is True
    )
    if not (source_ok or collection_source_ok):
        raise ValueError(
            "Expected a Cosmos3-Nano sort plan; run the perception pipeline first"
        )
    if not np.isclose(plan.get("open_gap_m", 0), 2 * C.JAW_OPEN):
        raise ValueError(
            "Plan uses a different jaw opening; regenerate capture and perception"
        )
    if plan.get("task") == "cube_stack" and plan["picks"]:
        if [p.get("colour") for p in plan["picks"]] != list(C.CUBE_STACK_COLOURS):
            raise ValueError(
                "Tower must contain two yellow, two blue, then two red cubes"
            )
        if len({p["detection_id"] for p in plan["picks"]}) != len(C.CUBE_STACK_COLOURS):
            raise ValueError("Tower repeats a cube detection")
        for level, p in enumerate(plan["picks"]):
            yaw = float(p.get("yaw_deg", float("nan")))
            theta = np.deg2rad(yaw)
            expected = [-np.sin(theta), np.cos(theta), 0.0]
            spread = np.asarray(p.get("spread_dir", []), float)
            destination = np.asarray(p.get("place_spread_dir", []), float)
            if (
                not np.isfinite(yaw)
                or not np.isfinite(float(p.get("pose_confidence", 0)))
                or float(p.get("pose_confidence", 0)) < 0.8
                or spread.shape != (3,)
                or not np.allclose(spread, expected)
                or destination.shape != (3,)
                or not np.allclose(destination, [0.0, 1.0, 0.0])
            ):
                raise ValueError(
                    "Cube grasp requires a confident pose and aligned wrist"
                )
            if p["category"] != "cube" or p.get("stack_level") != level:
                raise ValueError("Invalid cube tower level")
            if not np.allclose(p["place_xy"], C.CUBE_STACK_XY) or not np.isclose(
                p["place_z"], C.SURFACE_DZ + level * C.CUBE_EDGE + C.CUBE_TIP_CLEARANCE
            ):
                raise ValueError("Invalid tower destination or placement height")
            if not np.isclose(p.get("pick_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
                raise ValueError(
                    "Cube grasp height differs from the current hooked-tip setup"
                )
            if not np.isclose(
                p.get("expected_centre_z", -1),
                C.SURFACE_DZ + (level + 0.5) * C.CUBE_EDGE,
            ):
                raise ValueError("Invalid expected tower height")
    for p in plan["picks"]:
        if p["category"] == "cap":
            if (
                p.get("orientation") != "narrow_end_up"
                or float(p.get("orientation_confidence", 0))
                < C.CAP_ORIENTATION_MIN_CONFIDENCE
            ):
                raise ValueError(
                    "Plan contains a cap without a confident narrow-end-up observation"
                )
            if not np.isclose(p["pick_z"], C.SURFACE_DZ + C.CAP_PICK_HEIGHT):
                raise ValueError("Cap pick height must clear the wide flange")
        if p["category"] == "vial":
            axis = np.array(p.get("axis_world", []), float)
            spread = np.array(p.get("spread_dir", []), float)
            if (
                axis.shape != (2,)
                or spread.shape != (3,)
                or not np.isclose(np.linalg.norm(spread), 1)
                or abs(np.dot(axis, spread[:2])) > 1e-5
            ):
                raise ValueError("Tube must have a perpendicular jaw closing direction")
        for key in ("xy", "place_xy"):
            if len(p[key]) != 2 or not np.all(np.isfinite(p[key])):
                raise ValueError("Invalid plan coordinates")


def wide_up_object_names(run):
    """Return evaluation-only wide-up objects when ground truth is available.

    Dataset collection can legitimately start from a portable run containing
    only ``scene_state.json`` and ``sort_plan.json``.  Missing ground truth
    should omit the optional wide-cap evaluation, not discard an otherwise
    complete simulation report after every grasp has run.
    """
    path = Path(run) / "ground_truth.json"
    if not path.is_file():
        return []
    ground_truth = json.loads(path.read_text())
    return [
        obj["name"]
        for obj in ground_truth.get("objects", [])
        if obj.get("orientation") == "wide_end_up" and obj.get("name")
    ]


def main(argv=None, *, simulation_app=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        required=True,
        help="directory containing sort_plan.json and scene_state.json",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--sim-speedup",
        type=float,
        default=1.0,
        help=(
            "simulation-only motion playback multiplier; preserves the same "
            "waypoints and verification gates (0.25 to 3.0)"
        ),
    )
    ap.add_argument("--camera", choices=["table", "side", "wrist"], default="table")
    ap.add_argument(
        "--physics-only",
        action="store_true",
        help="validate physical grasps without rendering frames",
    )
    ap.add_argument(
        "--closed-loop",
        action="store_true",
        help="wrist grasp_check and place_check with bounded regrasp",
    )
    ap.add_argument(
        "--verification",
        choices=["rules", "ai", "ai_guarded", "stub"],
        default=None,
        help=(
            "grasp verification backend (default rules; ai modes need trained "
            "model; ai_guarded also requires rule agreement)"
        ),
    )
    ap.add_argument(
        "--trial-config",
        default=None,
        help="JSON trial from configs/trials (perturbation + policy)",
    )
    ap.add_argument(
        "--collect-dataset",
        default=None,
        metavar="DIR",
        help=(
            "Isaac: save table-camera before/after grasp crops; weak label at lift "
            "into DIR/manifest.json"
        ),
    )
    ap.add_argument(
        "--grasp-feedback",
        action="store_true",
        help="Append bad/mismatch grasp wrist frames to data/grasp_quality for retraining",
    )
    ap.add_argument(
        "--max-picks",
        type=int,
        default=None,
        help="Run only the first N planned picks (quick simulator validation)",
    )
    ap.add_argument(
        "--pick-index",
        type=int,
        default=None,
        help="Run only one zero-based planned pick (targeted data collection)",
    )
    ap.add_argument(
        "--rl-checkpoint",
        default=None,
        help="Q-table checkpoint for simulation-only policy='rl' trials",
    )
    ap.add_argument(
        "--rl-vision-config",
        default=None,
        help=(
            "before/after pickup-classifier config used as a fail-closed gate "
            "before an RL placement"
        ),
    )
    ap.add_argument(
        "--rl-vision-observe-only",
        action="store_true",
        help=(
            "score the before/after pickup classifier without allowing its gate "
            "to stop execution (benchmarking only)"
        ),
    )
    a = ap.parse_args(argv)
    if a.fps != 30:
        ap.error("Use 30 fps: two 1/60 s physics steps per video frame")
    if not 0.25 <= a.sim_speedup <= 3.0:
        ap.error("--sim-speedup must be between 0.25 and 3.0")
    if a.collect_dataset and a.physics_only:
        ap.error("--collect-dataset needs rendering; remove --physics-only")
    run = Path(a.run).resolve()
    out = Path(a.out).resolve() if a.out else run / "frames"
    if out.exists() and any(out.glob("*.png")):
        raise SystemExit(
            f"Output already contains PNGs: {out}. Choose a fresh run directory."
        )
    repo_root = ensure_stacking_recovery_on_path()
    trial_overrides = {}
    if a.trial_config:
        from stacking_recovery.bench.perturbations import trial_to_overrides
        from stacking_recovery.bench.trial_runtime import (
            apply_plan_overrides,
            load_trial_file,
        )

        trial_overrides = trial_to_overrides(
            load_trial_file(Path(a.trial_config).resolve())
        )
    plan = json.loads((run / "sort_plan.json").read_text())
    if trial_overrides:
        plan = apply_plan_overrides(plan, trial_overrides)
    validate_plan(plan, allow_collection_ground_truth=bool(a.collect_dataset))
    policy_mode = plan.get("closed_loop_policy", "open_loop")
    place_offset = np.asarray(
        plan.get("trial_place_xy_offset_m") or [0.0, 0.0], dtype=float
    )
    if place_offset.shape != (2,) or not np.all(np.isfinite(place_offset)):
        raise ValueError("place_xy_offset_m must contain two finite values")
    rl_adapter = None
    rl_visual_model = None
    if policy_mode == "rl":
        if not a.rl_checkpoint:
            ap.error("policy='rl' requires --rl-checkpoint")
        from stacking_recovery.integration.isaac_rl import IsaacRLPlacementAdapter

        rl_adapter = IsaacRLPlacementAdapter(Path(a.rl_checkpoint).resolve())
    if a.rl_vision_config:
        from stacking_recovery.vision.grasp_quality_model import (
            GraspQualityModel,
            load_grasp_config,
        )

        vision_config_path = Path(a.rl_vision_config).resolve()
        if not vision_config_path.is_file():
            ap.error(f"RL vision config not found: {vision_config_path}")
        rl_visual_model = GraspQualityModel(load_grasp_config(vision_config_path))
    for pick in plan["picks"]:
        target_xy = np.asarray(pick["place_xy"], dtype=float)
        if rl_adapter is not None:
            rl_decision = rl_adapter.decide(
                target_xy=target_xy.tolist(),
                injected_offset_m=place_offset.tolist(),
                stack_level=int(pick.get("stack_level", 0)),
            )
            pick["command_place_xy"] = list(rl_decision.commanded_xy)
            pick["rl_placement_decision"] = rl_decision.to_dict()
        else:
            pick["command_place_xy"] = (target_xy + place_offset).tolist()
    closed_loop = a.closed_loop or policy_mode in ("closed_loop", "rl")
    verification = (
        a.verification
        or trial_overrides.get("verification")
        or plan.get("verification")
        or "rules"
    )
    if trial_overrides.get("tip_static_friction") is not None:
        C.SOFT_TIP_STATIC_FRIC = float(trial_overrides["tip_static_friction"])
    if trial_overrides.get("finger") == "cube_flat_pad":
        C.SCALLOP_R = 0.001
    if not plan["picks"]:
        raise SystemExit(
            "No eligible picks; inspect detections.json and sort_plan.json before rendering"
        )
    import os

    grasp_feedback = a.grasp_feedback or os.environ.get("GRASP_FEEDBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    grasp_recorder = None
    if repo_root is not None and (grasp_feedback or a.collect_dataset):
        from stacking_recovery.data.dataset_split import split_for_seed
        from stacking_recovery.data.grasp_feedback import GraspFeedbackRecorder
        from stacking_recovery.vision.grasp_quality_model import load_grasp_config

        cfg = load_grasp_config(repo_root / "configs" / "grasp_model.yaml")
        fb = dict(cfg.get("feedback") or {})
        fb["enabled"] = True
        if a.collect_dataset:
            fb["mode"] = "collect_all"
            fb["data_root"] = str(Path(a.collect_dataset).resolve())
        else:
            fb["mode"] = "feedback"
        cfg["feedback"] = fb
        grasp_recorder = GraspFeedbackRecorder.from_config(repo_root, cfg)
        collection_seed = int(plan.get("trial_seed") or 0)
        collection_meta = {
            "seed": collection_seed,
            "split": split_for_seed(collection_seed),
            "trial_condition": plan.get("trial_condition") or "nominal",
            "injected_yaw_error_deg": float(plan.get("trial_yaw_offset_deg") or 0),
            "injected_pick_xy_offset_m": list(
                plan.get("trial_pick_xy_offset_m") or [0.0, 0.0]
            ),
            "campaign_target": plan.get("campaign_target"),
            "campaign_scenario": plan.get("campaign_scenario"),
            "run_id": run.name,
        }
    else:
        collection_meta = {}
    state = json.loads((run / "scene_state.json").read_text())
    # Resolve paths and validate before Kit changes the working directory.
    from isaacsim import SimulationApp

    owns_app = simulation_app is None
    app = simulation_app or SimulationApp({"headless": True})
    try:
        import cv2
        import omni.usd
        from pxr import UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell

        if not owns_app:
            # A batch worker keeps Kit alive but every trial still gets a new
            # USD stage and a fresh World singleton.  This prevents robot,
            # object, camera, and physics state from leaking between seeds.
            World.clear_instance()
            omni.usd.get_context().new_stage()
            app.update()

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
        stage = omni.usd.get_context().get_stage()
        cell.build(stage)
        wrappers = {}
        for name, saved in state["objects"].items():
            prim = stage.GetPrimAtPath("/World/" + name)
            if not prim.IsValid():
                raise ValueError(f"Captured object missing: {name}")
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
            wrapper = SingleXFormPrim("/World/" + name)
            wrapper.set_world_pose(
                np.array(saved["position"]), np.array(saved["orientation"])
            )
            wrappers[name] = wrapper
        cam_paths = {
            "table": cell.TABLE_CAM_PATH,
            "side": cell.SIDE_CAM_PATH,
            "wrist": cell.WRIST_CAM_PATH,
        }
        configs = {"table": C.TABLE_CAM, "side": C.SIDE_CAM, "wrist": C.WRIST_CAM}
        render_enabled = not a.physics_only
        record_video = render_enabled and not a.collect_dataset
        cam = (
            None
            if not record_video
            else Camera(
                prim_path=cam_paths[a.camera],
                resolution=configs[a.camera]["resolution"],
            )
        )
        cell.enable_ccd(world)
        world.reset()
        if cam is not None:
            cam.initialize()
        wrist_cam = None
        if closed_loop and render_enabled:
            wrist_cam = Camera(
                prim_path=cell.WRIST_CAM_PATH,
                resolution=C.WRIST_CAM["resolution"],
            )
            wrist_cam.initialize()
        table_observation_cam = None
        table_observation_model = None
        if (grasp_recorder is not None or rl_visual_model is not None) and render_enabled:
            if cam is not None and a.camera == "table":
                table_observation_cam = cam
            else:
                table_observation_cam = Camera(
                    prim_path=cell.TABLE_CAM_PATH,
                    resolution=C.TABLE_CAM["resolution"],
                )
                table_observation_cam.initialize()

            from backproject import CameraModel

            table_prim = stage.GetPrimAtPath(cell.TABLE_CAM_PATH)
            table_gcam = UsdGeom.Camera(table_prim)
            table_xf = UsdGeom.Xformable(table_prim).ComputeLocalToWorldTransform(0)
            table_w, table_h = C.TABLE_CAM["resolution"]
            table_observation_model = CameraModel(
                {
                    "resolution": [table_w, table_h],
                    "focal_length_mm": float(table_gcam.GetFocalLengthAttr().Get()),
                    "horizontal_aperture_mm": float(
                        table_gcam.GetHorizontalApertureAttr().Get()
                    ),
                    "vertical_aperture_mm": float(
                        table_gcam.GetVerticalApertureAttr().Get()
                    ),
                    "camera_to_world": [
                        [float(table_xf[r][c]) for c in range(4)] for r in range(4)
                    ],
                    "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
                }
            )
        recovery_policy = None
        RecoveryDecision = None
        ManipulationContext = None
        if closed_loop and repo_root is not None:
            from stacking_recovery.control.recovery_policy import (
                ManipulationContext as MC,
                RecoveryDecision as RD,
                RecoveryPolicy,
            )
            from stacking_recovery.vision.grasp_check_factory import get_grasp_check_fn
            from stacking_recovery.vision.place_check import place_check

            RecoveryDecision = RD
            ManipulationContext = MC
            recovery_policy = RecoveryPolicy(
                log_path=run / "manipulation.jsonl",
                grasp_check_fn=get_grasp_check_fn(verification),
                place_check_fn=place_check,
            )
        arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
        idx = {name: i for i, name in enumerate(names)}
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
        home = np.array([0.30, 0.0, 0.24])
        down = [0.0, 0.0, -1.0]
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q, ok = ik.solve(q, home, down, limits, tool_offset=tool)
        if not ok:
            raise RuntimeError("Home IK failed")

        def command(q6, jaw):
            full = arm.get_joint_positions().copy()
            full[arm_idx] = q6
            for joint, value in cell.jaw_targets(jaw).items():
                full[idx[joint]] = value
            arm.apply_action(ArticulationAction(joint_positions=full))

        full = arm.get_joint_positions().copy()
        full[arm_idx] = q
        for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
            full[idx[joint]] = value
        arm.set_joint_positions(full)
        for _ in range(120):
            command(q, C.JAW_OPEN)
            world.step(render=render_enabled)
        for name in wrappers:
            UsdPhysics.RigidBodyAPI(
                stage.GetPrimAtPath("/World/" + name)
            ).CreateKinematicEnabledAttr().Set(False)
        for _ in range(30):
            command(q, C.JAW_OPEN)
            world.step(render=render_enabled)

        def poses():
            values = {}
            for name, wrapper in wrappers.items():
                pos, quat = wrapper.get_world_pose()
                w, x, y, z = quat
                rot = np.array(
                    [
                        [
                            1 - 2 * (y * y + z * z),
                            2 * (x * y - z * w),
                            2 * (x * z + y * w),
                        ],
                        [
                            2 * (x * y + z * w),
                            1 - 2 * (x * x + z * z),
                            2 * (y * z - x * w),
                        ],
                        [
                            2 * (x * z - y * w),
                            2 * (y * z + x * w),
                            1 - 2 * (x * x + y * y),
                        ],
                    ]
                )
                centre = np.array(pos)
                if name.startswith("SortVial"):
                    centre += rot @ np.array([0, 0, C.VIAL_LENGTH / 2])
                elif name.startswith("SortCap"):
                    centre += rot @ np.array([0, 0, C.CAP_HEIGHT / 2])
                elif name.startswith("SortCube"):
                    centre += rot @ np.array([0, 0, C.CUBE_EDGE / 2])
                values[name] = {
                    "position": pos.tolist(),
                    "orientation": quat.tolist(),
                    "centre": centre.tolist(),
                    "yaw_deg": float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
                    "tilt_deg": float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))),
                }
            return values

        before = poses()
        events = []
        frame = 0
        abort_reason = None
        out.mkdir(parents=True, exist_ok=True)
        # Additional fixed side camera provides a clear lift/contact review.
        side = None
        if record_video and a.camera != "side":
            side = Camera(
                prim_path=cell.SIDE_CAM_PATH, resolution=C.SIDE_CAM["resolution"]
            )
            side.initialize()
        side_out = run / "side_frames"
        side_out.mkdir(exist_ok=True)
        if a.pick_index is not None:
            if a.pick_index < 0 or a.pick_index >= len(plan["picks"]):
                raise ValueError(f"--pick-index must be between 0 and {len(plan['picks']) - 1}")
            picks_to_run = [plan["picks"][a.pick_index]]
        else:
            picks_to_run = plan["picks"][: a.max_picks] if a.max_picks else plan["picks"]
        for pick in picks_to_run:
            pick_started_at = time.perf_counter()
            # Select equivalent jaw-direction signs once for the whole grasp,
            # so approach/descend and carry/lower cannot differ by 180 degrees.
            pick = dict(pick)
            if pick.get("spread_dir") is not None:
                r, _ = fk.fk(q)
                current = r @ np.array(C.FINGER_SPREAD_AXIS)
                spread = np.array(pick["spread_dir"], float)
                if np.dot(current, spread) < 0:
                    spread = -spread
                pick["spread_dir"] = spread.tolist()
                if pick.get("place_spread_dir") is not None:
                    destination = np.array(pick["place_spread_dir"], float)
                    if np.dot(spread, destination) < 0:
                        destination = -destination
                    pick["place_spread_dir"] = destination.tolist()
            start = poses()
            collection_before_bgr = None
            collection_observation_meta = {}
            if table_observation_cam is not None and table_observation_model is not None:
                table_rgba = table_observation_cam.get_rgba()
                if table_rgba is None:
                    raise RuntimeError("Table camera returned no before-grasp image")
                collection_before_bgr = rgba_to_bgr(table_rgba)
                observation_xy = pick.get("observation_xy", pick["xy"])
                crop_uv = table_observation_model.project(
                    [
                        float(observation_xy[0]),
                        float(observation_xy[1]),
                        C.TABLE_TOP_Z + C.SURFACE_DZ + C.CUBE_EDGE / 2,
                    ]
                )
                if crop_uv is None:
                    raise RuntimeError("Planned pick is behind the table camera")
                collection_observation_meta = {
                    "observation_source": "table_before_after",
                    "crop_center_uv": [float(crop_uv[0]), float(crop_uv[1])],
                    "crop_size_px": 128,
                    "camera_resolution": [table_w, table_h],
                }
            prefix = {"cap": "SortCap", "cube": "SortCube", "vial": "SortVial"}[
                pick["category"]
            ]
            candidates = [name for name in wrappers if name.startswith(prefix)]
            # This association is for execution reporting only, never motion commands.
            tracked = (
                min(
                    candidates,
                    key=lambda name: np.linalg.norm(
                        np.array(start[name]["centre"][:2]) - pick["xy"]
                    ),
                )
                if candidates
                else None
            )
            if (
                tracked
                and np.linalg.norm(np.array(start[tracked]["centre"][:2]) - pick["xy"])
                > 0.04
            ):
                tracked = None
            event = {
                "detection_id": pick["detection_id"],
                "category": pick["category"],
                "tracked_object": tracked,
                "pickup_verified": False,
                "placement_verified": False,
                "trajectory_ik_failures": 0,
                "target_place_xy": list(pick["place_xy"]),
                "command_place_xy": list(pick.get("command_place_xy", pick["place_xy"])),
            }
            if pick.get("rl_placement_decision") is not None:
                event["rl_placement_decision"] = pick["rl_placement_decision"]
                if pick["rl_placement_decision"].get("safe_stop"):
                    event["rl_safe_stop"] = True
                    events.append(event)
                    abort_reason = (
                        f'RL safety stop before cube level {pick.get("stack_level")}'
                    )
                    print(f"[sort] {abort_reason}", flush=True)
                    break
            if "stack_level" in pick:
                event.update(
                    stack_level=pick["stack_level"],
                    colour=pick["colour"],
                    perceived_yaw_deg=pick["yaw_deg"],
                    grasp_spread_dir=pick["spread_dir"],
                    place_spread_dir=pick["place_spread_dir"],
                )
            event["initial_positions"] = start
            event["closed_loop_enabled"] = bool(recovery_policy)
            max_grasp_retries = (
                recovery_policy.MAX_GRASP_RETRIES if recovery_policy else 0
            )
            max_place_retries = (
                recovery_policy.MAX_PLACEMENT_RETRIES if recovery_policy else 0
            )
            placement_attempt = 0
            pick_finished = False
            pick_abort = False
            closed_loop_ctx = None
            while not pick_finished and placement_attempt <= max_place_retries:
                grasp_attempt = 0
                retry_grasp = True
                while retry_grasp and grasp_attempt <= max_grasp_retries:
                    retry_grasp = False
                    for name, p0, p1, duration, j0, j1, spread in segments(
                        pick, home
                    ):
                        spread_start = None
                        if spread is not None and name in ("approach", "carry"):
                            r, _ = fk.fk(q)
                            s = r @ np.array(C.FINGER_SPREAD_AXIS)
                            s[2] = 0
                            s /= np.linalg.norm(s)
                            spread = np.array(spread, float)
                            spread_start = s
                        # Keep the exact same Cartesian path and all safety
                        # checks while using fewer interpolation frames for a
                        # faster digital-twin benchmark.  This flag is not a
                        # real-robot speed command.
                        n = max(2, round(duration * a.fps / a.sim_speedup))
                        for i in range(n):
                            t = ease((i + 1) / n)
                            target = p0 + (p1 - p0) * t
                            direction = (
                                turn(spread_start, spread, t)
                                if spread_start is not None
                                else spread
                            )
                            q, ok = ik.solve(
                                q, target, down, limits, tool_offset=tool, spread_dir=direction
                            )
                            if not ok:
                                q, ok = ik.solve(
                                    q,
                                    target,
                                    down,
                                    limits,
                                    tool_offset=tool,
                                    spread_dir=direction,
                                    iters=500,
                                    damping=0.02,
                                )
                            r, actual = fk.tool_pose(q, tool)
                            aligned = (
                                direction is None
                                or abs(np.dot(r @ np.array(C.FINGER_SPREAD_AXIS), direction))
                                > 0.999
                            )
                            if not ok or np.linalg.norm(actual - target) > 0.001 or not aligned:
                                event["trajectory_ik_failures"] += 1
                                raise RuntimeError(
                                    f'IK failed: detection {pick["detection_id"]}, {name}, frame {i}; refusing off-target descent'
                                )
                            command(q, j0 + (j1 - j0) * t)
                            for _ in range(2):
                                world.step(render=render_enabled)
                            if cam is not None:
                                rgba = cam.get_rgba()
                                if rgba is None:
                                    raise RuntimeError("Camera returned no image")
                                cv2.imwrite(
                                    str(out / f"f{frame:05d}.png"),
                                    cv2.cvtColor(
                                        rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                                    ),
                                )
                            if side is not None:
                                rgba_side = side.get_rgba()
                                if rgba_side is not None:
                                    cv2.imwrite(
                                        str(side_out / f"f{frame:05d}.png"),
                                        cv2.cvtColor(
                                            rgba_side[:, :, :3].astype(np.uint8),
                                            cv2.COLOR_RGB2BGR,
                                        ),
                                    )
                            frame += 1
                        if name in (
                            "descend",
                            "dwell",
                            "lift",
                            "lower",
                            "prelower",
                            "final_lower",
                            "release",
                        ):
                            measured = arm.get_joint_positions()
                            rot, tip = fk.tool_pose(measured[arm_idx], tool)
                            measured_spread = rot @ np.array(C.FINGER_SPREAD_AXIS)
                            event.setdefault("phase_measurements", {})[name] = {
                                "actual_tip_robot_frame_m": tip.tolist(),
                                "actual_spread_dir": measured_spread.tolist(),
                                "commanded_spread_dir": (
                                    np.asarray(direction).tolist()
                                    if direction is not None
                                    else None
                                ),
                                "wrist_joint_deg": float(np.degrees(measured[arm_idx][-1])),
                                "commanded_tip_robot_frame_m": target.tolist(),
                                "tip_error_mm": float(np.linalg.norm(tip - target) * 1000),
                                "object_centre_world_m": (
                                    poses()[tracked]["centre"] if tracked else None
                                ),
                            }
                        if name == "dwell":
                            jp = arm.get_joint_positions()
                            event["jaw_gap_mm"] = float(
                                (2 * C.JAW_OPEN - abs(jp[idx["jaw_l"]]) - abs(jp[idx["jaw_r"]]))
                                * 1000
                            )
                            grasp_bgr = None
                            grasp_result = None
                            if wrist_cam is not None and (
                                recovery_policy is not None or grasp_recorder is not None
                            ):
                                for _ in range(5):
                                    world.step(render=render_enabled)
                                grasp_bgr = rgba_to_bgr(wrist_cam.get_rgba())
                            if recovery_policy is not None and grasp_bgr is not None:
                                closed_loop_ctx = ManipulationContext(
                                    cube_colour=pick.get("colour", "unknown"),
                                    stack_level=int(pick.get("stack_level", 0)),
                                    table_pose_confidence=float(
                                        pick.get("pose_confidence", 0.9)
                                    ),
                                    grasp_attempt_index=grasp_attempt,
                                    placement_attempt_index=placement_attempt,
                                )
                                decision, grasp_result = recovery_policy.after_grasp_close(
                                    closed_loop_ctx, grasp_bgr
                                )
                                event["closed_loop_grasp"] = grasp_result.to_dict()
                                event["closed_loop_grasp_looks"] = [
                                    grasp_result.to_dict()
                                ]
                                while decision == RecoveryDecision.SECOND_LOOK:
                                    closed_loop_ctx.uncertain_look_index += 1
                                    for _ in range(5):
                                        world.step(render=render_enabled)
                                    grasp_bgr = rgba_to_bgr(wrist_cam.get_rgba())
                                    decision, grasp_result = (
                                        recovery_policy.after_grasp_close(
                                            closed_loop_ctx, grasp_bgr
                                        )
                                    )
                                    event["closed_loop_grasp"] = (
                                        grasp_result.to_dict()
                                    )
                                    event["closed_loop_grasp_looks"].append(
                                        grasp_result.to_dict()
                                    )
                                if grasp_recorder is not None:
                                    grasp_recorder.on_dwell(
                                        collection_before_bgr
                                        if collection_before_bgr is not None
                                        else grasp_bgr,
                                        grasp_result,
                                        detection_id=pick.get("detection_id"),
                                        stack_level=pick.get("stack_level"),
                                        **collection_observation_meta,
                                        **collection_meta,
                                    )
                                if decision in (
                                    RecoveryDecision.REGRASP,
                                    RecoveryDecision.REACQUIRE,
                                ):
                                    if decision == RecoveryDecision.REACQUIRE:
                                        event["table_reacquire_requested"] = True
                                    grasp_attempt += 1
                                    retry_grasp = True
                                    break
                                if decision == RecoveryDecision.SAFE_STOP:
                                    if grasp_recorder is not None:
                                        grasp_recorder.on_grasp_failure_no_lift(
                                            grasp_bgr,
                                            grasp_result,
                                            detection_id=pick.get("detection_id"),
                                            stack_level=pick.get("stack_level"),
                                        )
                                    pick_abort = True
                                    abort_reason = (
                                        f'Closed-loop grasp stop at level {pick.get("stack_level")}: '
                                        f"{grasp_result.reason.value}"
                                    )
                                    break
                            elif grasp_recorder is not None and collection_before_bgr is not None:
                                from stacking_recovery.types import (
                                    FailureReason,
                                    GraspCheckResult,
                                )

                                grasp_result = GraspCheckResult(
                                    seated=True,
                                    yaw_error_deg=0.0,
                                    confidence=1.0,
                                    reason=FailureReason.STUB,
                                    verification="collect",
                                )
                                grasp_recorder.on_dwell(
                                    collection_before_bgr,
                                    grasp_result,
                                    detection_id=pick.get("detection_id"),
                                    stack_level=pick.get("stack_level"),
                                    **collection_observation_meta,
                                    **collection_meta,
                                )
                        if name == "settle" and recovery_policy is not None:
                            verify_cam = cam if cam is not None else wrist_cam
                            if verify_cam is not None and closed_loop_ctx is not None:
                                place_bgr = rgba_to_bgr(verify_cam.get_rgba())
                                decision, place_result = recovery_policy.after_place(
                                    closed_loop_ctx, place_bgr
                                )
                                event["closed_loop_place"] = place_result.to_dict()
                                if decision == RecoveryDecision.REGRASP:
                                    placement_attempt += 1
                                    grasp_attempt = 0
                                    retry_grasp = True
                                    break
                                if decision == RecoveryDecision.SAFE_STOP:
                                    pick_abort = True
                                    abort_reason = (
                                        f'Closed-loop placement stop at level {pick.get("stack_level")}: '
                                        f"{place_result.reason.value}"
                                    )
                                    break
                        if pick_abort:
                            break
                        if name == "lift":
                            lifted = poses()
                            event["objects_lifted"] = [
                                key
                                for key in wrappers
                                if lifted[key]["centre"][2] - start[key]["centre"][2] > 0.025
                            ]
                            event["pickup_verified"] = tracked in event["objects_lifted"]
                            after_bgr = None
                            if table_observation_cam is not None:
                                table_rgba = table_observation_cam.get_rgba()
                                if table_rgba is not None:
                                    after_bgr = rgba_to_bgr(table_rgba)
                            if grasp_recorder is not None:
                                saved = grasp_recorder.on_lift(
                                    event, after_image_bgr=after_bgr
                                )
                                if saved:
                                    event["grasp_feedback_saved"] = saved
                            if rl_visual_model is not None:
                                from stacking_recovery.data.table_grasp_observation import (
                                    make_before_after_observation,
                                )
                                from stacking_recovery.integration.visual_rl_guard import (
                                    VisualGateAction,
                                    visual_pickup_gate,
                                )

                                if collection_before_bgr is None or after_bgr is None:
                                    raise RuntimeError(
                                        "RL visual verification requires table before/after images"
                                    )
                                center_uv = tuple(
                                    collection_observation_meta["crop_center_uv"]
                                )
                                crop_size = int(
                                    collection_observation_meta["crop_size_px"]
                                )
                                pair_bgr, pair_quality = make_before_after_observation(
                                    collection_before_bgr,
                                    after_bgr,
                                    center_uv,
                                    crop_size=crop_size,
                                )
                                predictions = []
                                if pair_quality.valid:
                                    first_prediction = rl_visual_model.predict(
                                        pair_bgr,
                                        observation_source="table_before_after",
                                    )
                                    predictions.append(first_prediction.to_dict())
                                    visual_gate = visual_pickup_gate(first_prediction)
                                else:
                                    visual_gate = None

                                if (
                                    visual_gate is not None
                                    and visual_gate.action
                                    == VisualGateAction.SECOND_LOOK
                                ):
                                    for _ in range(5):
                                        world.step(render=render_enabled)
                                    second_rgba = table_observation_cam.get_rgba()
                                    if second_rgba is None:
                                        raise RuntimeError(
                                            "Table camera returned no second-look image"
                                        )
                                    second_bgr = rgba_to_bgr(second_rgba)
                                    second_pair, second_quality = (
                                        make_before_after_observation(
                                            collection_before_bgr,
                                            second_bgr,
                                            center_uv,
                                            crop_size=crop_size,
                                        )
                                    )
                                    if second_quality.valid:
                                        second_prediction = rl_visual_model.predict(
                                            second_pair,
                                            observation_source="table_before_after",
                                        )
                                        predictions.append(second_prediction.to_dict())
                                        visual_gate = visual_pickup_gate(
                                            first_prediction, second_prediction
                                        )
                                    else:
                                        pair_quality = second_quality
                                        visual_gate = None

                                event["rl_visual_verification"] = {
                                    "quality": pair_quality.to_dict(),
                                    "predictions": predictions,
                                    "observe_only": bool(a.rl_vision_observe_only),
                                    "gate": visual_gate.to_dict()
                                    if visual_gate is not None
                                    else {
                                        "action": "safe_stop",
                                        "reason": pair_quality.reason
                                        or "invalid_second_look",
                                    },
                                }
                                if (
                                    not a.rl_vision_observe_only
                                    and (
                                        visual_gate is None
                                        or visual_gate.action
                                        == VisualGateAction.SAFE_STOP
                                    )
                                ):
                                    pick_abort = True
                                    abort_reason = (
                                        "RL visual safety gate rejected pickup at "
                                        f"level {pick.get('stack_level')}"
                                    )
                            print(
                                f'[sort] detection {pick["detection_id"]}: lifted {event["objects_lifted"]}; pickup={event["pickup_verified"]}',
                                flush=True,
                            )
                        if pick_abort:
                            break
                        if not retry_grasp:
                            pick_finished = True
                    if pick_abort:
                        break
                if pick_abort:
                    break
            event["closed_loop_grasp_attempts"] = grasp_attempt
            event["closed_loop_placement_attempts"] = placement_attempt
            event["cycle_time_s"] = time.perf_counter() - pick_started_at
            if pick_abort:
                events.append(event)
                print(f"[sort] {abort_reason}", flush=True)
                break
            finish = poses()
            if tracked:
                position = np.array(finish[tracked]["centre"])
                event["placement_error_mm"] = float(
                    np.linalg.norm(position[:2] - pick["place_xy"]) * 1000
                )
                expected_z = expected_placement_centre_z(pick["category"])
                if "stack_level" in pick:
                    expected_z = C.TABLE_TOP_Z + pick["expected_centre_z"]
                event["placement_verified"] = bool(
                    event["pickup_verified"]
                    and event["placement_error_mm"] < 20
                    and abs(position[2] - expected_z) < 0.01
                )
                if "stack_level" in pick:
                    event["placement_verified"] = bool(
                        event["pickup_verified"]
                        and event["placement_error_mm"]
                        < C.CUBE_STACK_SITE_TOLERANCE * 1000
                        and abs(position[2] - expected_z) < 0.0015
                    )
                    event["placement_yaw_error_deg_mod90"] = abs(
                        square_yaw(finish[tracked]["yaw_deg"])
                    )
                    event["placement_tilt_deg"] = finish[tracked]["tilt_deg"]
                    event["placement_pose_verified"] = bool(
                        event["placement_yaw_error_deg_mod90"] < 8
                        and event["placement_tilt_deg"] < 5
                    )
                    event["placement_verified"] = (
                        event["placement_verified"] and event["placement_pose_verified"]
                    )
                    event["stack_support_verified"] = verify_cube_stack(
                        finish, events + [event]
                    )
                    disturbance = measure_stack_disturbance(start, finish, events)
                    event.update(disturbance)
                    event["placement_verified"] = bool(
                        event["placement_verified"]
                        and disturbance["tower_disturbance_verified"]
                    )
            events.append(event)
            (run / "execution_events.json").write_text(
                json.dumps(events, indent=2) + "\n"
            )
            if plan.get("task") == "cube_stack" and (
                not event["placement_verified"]
                or not event.get("stack_support_verified")
            ):
                abort_reason = f'Cube tower stopped at level {pick["stack_level"]}: pickup, placement or supporting stack failed'
                print(f"[sort] {abort_reason}", flush=True)
                break
        after = poses()
        report = {
            "before": before,
            "after": after,
            "picks": events,
            "frames": frame,
            "rendered": render_enabled,
            "video_frames_saved": record_video,
            "task": plan.get("task", "sort"),
            "abort_reason": abort_reason,
            "hardware": {
                "gripper_model": C.GRIPPER_MODEL,
                "jaw_stroke_m": C.JAW_STROKE,
                "open_gap_m": 2 * C.JAW_OPEN,
                "finger_length_m": C.FINGER_LENGTH,
                "cube_edge_m": C.CUBE_EDGE,
                "cube_tip_clearance_m": C.CUBE_TIP_CLEARANCE,
            },
            "closed_loop": closed_loop,
            "policy": policy_mode,
            "grasp_verification": verification if closed_loop else None,
            "grasp_feedback": bool(grasp_recorder),
            "grasp_dataset_collect": bool(a.collect_dataset),
            "trial_condition": plan.get("trial_condition"),
            "finger_id": plan.get("finger_id"),
            "task_complete": bool(plan["picks"])
            and abort_reason is None
            and len(events) == len(plan["picks"])
            and all(e["placement_verified"] for e in events),
            "verified_pickups": sum(e["pickup_verified"] for e in events),
            "verified_placements": sum(e["placement_verified"] for e in events),
        }
        # Evaluate untouched wide-up objects only after executing the vision plan.
        wide = wide_up_object_names(run)
        report["wide_caps"] = [
            {
                "name": name,
                "displacement_mm": float(
                    np.linalg.norm(
                        np.array(after[name]["centre"]) - before[name]["centre"]
                    )
                    * 1000
                ),
                "picked_up": any(name in e.get("objects_lifted", []) for e in events),
            }
            for name in wide
        ]
        (run / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f'[sort] {report["verified_pickups"]}/{len(events)} pickups; {report["verified_placements"]} verified placements; {frame} frames',
            flush=True,
        )
    except Exception:
        # Kit shutdown may terminate Python before the exception is displayed.
        import traceback

        traceback.print_exc()
        raise
    finally:
        if owns_app:
            app.close()


if __name__ == "__main__":
    main()
