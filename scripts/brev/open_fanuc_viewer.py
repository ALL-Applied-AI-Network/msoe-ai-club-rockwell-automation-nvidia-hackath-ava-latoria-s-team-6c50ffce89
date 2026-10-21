"""Open the saved FANUC workcell and run a safe streamed motion demo."""

import asyncio

import numpy as np
import omni.kit.app
import omni.kit.viewport.utility
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


STAGE_PATH = "/workspace/team-data/fanuc_cell.usd"
CAMERA_PATH = "/World/TableCam"
ROBOT_PATH = "/World/ER4iA"

# These poses stay close to the validated home pose and well inside the FANUC
# joint limits.  They demonstrate control without reaching down toward objects.
POSES_DEG = (
    ("home", (0.0, 35.0, -25.0, 0.0, -80.0, 0.0), True),
    ("look-left", (-18.0, 31.0, -21.0, 12.0, -76.0, -22.0), True),
    ("gripper-close", (0.0, 27.0, -18.0, 0.0, -70.0, 0.0), False),
    ("look-right", (18.0, 39.0, -29.0, -12.0, -84.0, 22.0), True),
)


def smoothstep(value: float) -> float:
    """Ease a normalized value so each move starts and stops gently."""
    return value * value * (3.0 - 2.0 * value)


def jaw_targets(opened: bool) -> tuple[float, float]:
    """Return joint positions for the two 6 mm gripper slides."""
    return (0.0, 0.0) if opened else (-0.006, 0.006)


async def open_workcell() -> None:
    context = omni.usd.get_context()
    opened, error = await context.open_stage_async(STAGE_PATH)
    if not opened:
        raise RuntimeError(f"Could not open {STAGE_PATH}: {error}")

    # Let the streamed viewport observe the new stage before switching cameras.
    app = omni.kit.app.get_app()
    for _ in range(3):
        await app.next_update_async()

    viewport = omni.kit.viewport.utility.get_active_viewport()
    if viewport is not None:
        viewport.set_active_camera(CAMERA_PATH)

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    await world.initialize_simulation_context_async()
    await world.reset_async()

    arm = SingleArticulation(prim_path=ROBOT_PATH, name="streamed_er4ia")
    arm.initialize()
    names = list(arm.dof_names)
    required = [*(f"J{i}" for i in range(1, 7)), "jaw_l", "jaw_r"]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"FANUC articulation is missing joints: {missing}")

    indices = {name: names.index(name) for name in required}
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print(f"[fanuc-demo] stage loaded; DOFs={names}", flush=True)

    # Start exactly at the known home pose so reconnecting viewers never see a
    # sudden jump from an unknown saved articulation state.
    positions = arm.get_joint_positions().copy()
    home = np.deg2rad(np.asarray(POSES_DEG[0][1], dtype=float))
    for joint, value in zip(required[:6], home):
        positions[indices[joint]] = value
    positions[indices["jaw_l"]], positions[indices["jaw_r"]] = jaw_targets(True)
    arm.set_joint_positions(positions)

    current_arm = home
    current_jaw = np.asarray(jaw_targets(True), dtype=float)
    print("[fanuc-demo] safe looping demonstration started", flush=True)

    while True:
        for label, target_degrees, opened in POSES_DEG[1:] + POSES_DEG[:1]:
            target_arm = np.deg2rad(np.asarray(target_degrees, dtype=float))
            target_jaw = np.asarray(jaw_targets(opened), dtype=float)
            start_arm = current_arm.copy()
            start_jaw = current_jaw.copy()
            print(f"[fanuc-demo] moving to {label}", flush=True)

            # Three seconds per move at the nominal 60 Hz physics rate.
            for frame in range(180):
                amount = smoothstep((frame + 1) / 180.0)
                commanded = arm.get_joint_positions().copy()
                arm_values = start_arm + (target_arm - start_arm) * amount
                jaw_values = start_jaw + (target_jaw - start_jaw) * amount
                for joint, value in zip(required[:6], arm_values):
                    commanded[indices[joint]] = value
                commanded[indices["jaw_l"]] = jaw_values[0]
                commanded[indices["jaw_r"]] = jaw_values[1]
                arm.apply_action(ArticulationAction(joint_positions=commanded))
                await app.next_update_async()

            current_arm = target_arm
            current_jaw = target_jaw

            # Brief pause makes each pose and the gripper action easy to see.
            for _ in range(60):
                await app.next_update_async()


asyncio.ensure_future(open_workcell())
