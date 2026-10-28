# Hardware

Export printable finger geometry here (`cad/`). Parameters live in [`configs/finger/`](../configs/finger/).

SMC MHZL2-16D2: 6 mm/jaw, ~12 mm straight-face opening, 76 mm blade length (see starter `scene_config.py`).

## Safe software boundary

The repository now exposes a robot-neutral `RobotAdapter` with only five
operations: `move_to_pose`, `open_gripper`, `close_gripper`, `stop`, and
`get_status`.  The checked-in implementation is **shadow mode**: it records
commands but never connects to or moves a robot.

Start from [`configs/hardware/fanuc_schunk_template.json`](../configs/hardware/fanuc_schunk_template.json).
It intentionally contains no controller address, driver, I/O numbers, tool
frame, payload, or workspace limits.  Those values must come from the lab
mentor.  Selecting an unsupported real mode raises `HardwareInterlockError`
instead of guessing a connection.

The Isaac simulator continues to use its native articulation code.  A future
mentor-reviewed adapter should translate the same five operations to the
approved FANUC interface and SCHUNK handshake while retaining shadow mode for
observation-only qualification.

Generate a no-motion transcript from an existing plan before the lab:

```bash
python scripts/run_hardware_shadow.py \
  --plan /path/to/run/sort_plan.json \
  --out results/shadow_commands.json
```

The runner only accepts `mode: shadow`, validates every coordinate, limits the
recorded speed scale to 25%, and writes approach, pick, lift, place, retreat and
gripper operations.  It has no network or robot-driver code and reports
`motion_sent: false` in the transcript.

## Placement verification

`configs/placement_verification.json` contains the real-camera image-space
gate.  It checks component size, centre, aspect ratio, and expected support-line
position instead of accepting any saturated colour in a broad ROI.  Calibrate
the ROI, expected centre and per-level vertical shift from observation-only
lab images before allowing the result to gate motion.
