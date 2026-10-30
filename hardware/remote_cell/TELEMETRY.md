# Remote arm telemetry

The organizer records all three camera streams and robot state at 10 Hz. This
policy therefore does **not** copy full images during a physical run. Instead,
each `Policy.act()` call saves the policy's exact decision context beside the
organizer recording.

By default the files are written to:

`/workspace/team-data/remote-arm-telemetry/<session-id>/`

- `events.jsonl` — one record per policy decision
- `latest.json` — the newest record, useful for a live dashboard
- `summary.json` — current phase, outcome, safety result, and latency
- `../current_session.json` — points tools to the active session

Each event includes the state-machine phase before and after the decision,
red/blue detections, calibrated target positions, robot pose and joints, the
exact returned TCP actions, safety validation, outcome/reason, and policy
latency. It stores only camera names and frame shapes, not image pixels.

Language-directed and six-cube events also include `policy_name`, the
schema-validated `task_plan`, and a `six_cube` object with the active move,
tower position, planned move count, verified layers, and pickup-verification
source. The dashboard can show the selected skill and deterministic
state-machine progress from these fields without receiving authority to create
robot coordinates or motion commands.

Set `REMOTE_ARM_TELEMETRY_DIR` before starting the official client to choose a
different output folder. A logging failure is isolated from control: it cannot
modify an action or stop the policy.

After a remote run, use **My Results** or `uv run run.py --download` for the
organizer video. Match the downloaded timestamps with `wall_time_utc` in
`events.jsonl`; the React/TypeScript dashboard can poll `latest.json` during a
run and load the complete JSONL afterward.
