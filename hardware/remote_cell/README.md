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

The first physical policy intentionally does **not** use online RL. It detects
the red and blue cubes in the calibrated table view, executes a slow vertical
pick/place sequence, verifies the pickup using the table and wrist views, and
verifies the final red-on-blue placement. Any missing calibration, invalid
coordinate, missing camera, or failed verification stops the sequence and
holds the current pose.

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

