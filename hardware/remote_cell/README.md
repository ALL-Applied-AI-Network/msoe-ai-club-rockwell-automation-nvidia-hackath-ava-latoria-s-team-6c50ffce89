# Official remote cell policy

`policy.py` is a conservative adapter for the organizers' official
`PhysicalAI-Hackathon-2026` client. It uses that client's calibrated camera
geometry and TCP action API. It does not connect to the arm directly.

Before any physical rollout:

1. Copy `policy.py` over the official client's `policy.py`.
2. Run the official local mock. The mock checks the API and safety limits, not
   whether the vision logic can find real cubes.
3. Submit the official YAML and book a supervised slot.
4. Use the first rollout only to confirm camera colors, calibration, approach
   height, and gripper close direction. Keep the operator ready to stop.

The physical policy intentionally does **not** use online RL. Its validated
default path detects
the red and blue cubes in the calibrated table view, executes a slow vertical
pick/place sequence, verifies the pickup using the table and wrist views, and
verifies the final red-on-blue placement. Any missing calibration, invalid
coordinate, missing camera, or failed verification stops the sequence and
holds the current pose.

## Supervised six-cube mode

`SafeSixCubeStackV1` is a separate, explicitly enabled path for an
operator-confirmed layout containing exactly two yellow, two blue, and two red
10 mm cubes. One yellow cube remains on the table as the tower base; the policy
then makes five verified placements in yellow, blue, blue, red, red order. It
raises the placement target by 10 mm per layer, maintains a vertical approach
and retreat, checks every generated TCP target, verifies pickup disappearance
and the top color at the tower site, and stops on any ambiguity.

Enable it only for a supervised six-cube trial:

```bash
export REMOTE_ARM_STACK_ALL_CUBES=1
```

The same mode is selected automatically by an unambiguous instruction such as
`Stack all six cubes into one tower.` The normal red-on-blue instruction still
uses `SafeRedOnBlueV1`, and leaving the environment variable unset preserves
the physically validated behavior. Six-cube mode requires the calibrated table
camera; the single-position operator fallback is deliberately rejected because
it cannot identify duplicate colors. Before a physical rollout, confirm the
six cubes are separated by at least 25 mm, the full tower will remain inside
the robot workspace, and the operator is ready to stop the first trial.

## Supervised calibration fallback

If the operator has independently confirmed the two cube centers in robot-base
millimeters, a supervised run may provide both targets explicitly:

```bash
export REMOTE_ARM_RED_XY_MM="420,-100"
export REMOTE_ARM_BLUE_XY_MM="500,100"
```

Both variables are required. The policy rejects malformed, non-finite,
out-of-bounds, or out-of-reach targets before connecting. The fallback is used
only when the requested color is visible in the table image but its calibrated
position fails the normal TCP/reach checks. Wrist-camera pickup verification
still runs. Since a broken table calibration cannot independently verify the
final placement, telemetry records `completed_unverified` and the organizer's
camera/scoring remains the source of truth.

This is a supervised recovery path for a known physical layout, not a general
replacement for camera-to-robot calibration. Arbitrary layouts must have a
valid calibration (or an equivalent trusted pose source).

The official client supplies `obs["images"]["table"]` as RGB. The table image
is used upright (robot base at the top, robot `+x` down the image, and robot
`+y` to the right); this policy does not rotate, mirror, or swap RGB/BGR
channels.

## Optional Gemini language planner

The policy reads the organizer's `obs["trial"]["instruction"]` before any
motion. The known benchmark instruction is parsed locally, so the physical run
does not depend on network access or an LLM. Other wording can optionally be
converted to a strict task schema by Gemini:

```bash
export GEMINI_API_KEY="your-key-from-Google-AI-Studio"
export GEMINI_MODEL="gemini-2.5-flash-lite"   # optional override
export GEMINI_TIMEOUT_SECONDS="5"             # clamped to 1-15 seconds
```

Never paste the key into `policy.py`, a YAML file, telemetry, or Git. The key is
sent only in the `x-goog-api-key` request header and is not written to logs.

Gemini may select only a structured, allowlisted task. It cannot produce TCP
coordinates, joint angles, motion commands, or executable code. Separate strict
schemas cover the validated red-on-blue skill and `SafeSixCubeStackV1`; the
six-cube schema requires exactly six 10 mm cubes and the fixed yellow, yellow,
blue, blue, red, red order. The physical policy still resolves the known
instructions locally, so a rollout does not depend on Gemini or network access.
The dashboard can force Gemini planning for a preview, validate the returned
schema, and then pass only the selected allowlisted skill to the state machine.
Any ambiguous instruction, unsupported object/action, invalid response, timeout,
or network error stops before motion. Future language-directed skills should be
added as separately tested state-machine handlers rather than giving the LLM
direct robot control.

