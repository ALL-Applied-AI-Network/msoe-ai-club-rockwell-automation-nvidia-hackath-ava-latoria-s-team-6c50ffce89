# One-Day FANUC Lab Readiness

The lab goal is not to tune a model on the robot. The goal is to arrive with
one frozen release, prove each assumption at reduced speed, and then run the
full tower. A trained FANUC operator must supervise every powered movement and
own the controller, safety perimeter, reduced-speed mode, and emergency stop.

## What must be frozen before the visit

- One Git commit/tag containing the deterministic fallback and the RL option.
- The exact Q-table, classifier checkpoint, configs, cube order, TCP, gripper
  geometry, camera intrinsics/extrinsics, and table-to-robot transform.
- A local/offline copy of every dependency and model. Run `git lfs pull` and
  confirm the classifier file is a real model, not a 133-byte LFS pointer.
- A printed recovery plan: uncertain grasp means second look; rejected grasp
  means no carry to the stack; exhausted retries mean supervised safe stop.
- No online training and no changing thresholds during the lab demonstration.

## Evidence required this week

Do not call the system lab-ready until all of these are recorded:

1. **Independent classifier holdout:** at least 250 test pairs, at least 95%
   accuracy, and at most 2% unsafe `GOOD` predictions. The new external test
   campaign is the correct dataset for this gate.
2. **Matched Isaac comparison:** open-loop, deterministic closed-loop, and
   visual-guarded RL use the same scenes. Run at least 20 full six-cube trials
   per policy. Select a policy only from these results, not the lightweight RL
   environment.
3. **Nominal tower reliability:** the selected policy completes at least 18 of
   20 full towers, with zero unsafe motions and zero ignored verification
   failures.
4. **Failure handling:** at least 20 intentional failures across missed grasp,
   shifted cube, occlusion, and placement offset. The robot must retry, reject,
   or stop as designed every time; it must never carry a classifier-rejected
   grasp toward the tower.
5. **Repeatability:** three clean simulator restarts produce the same plan and
   successful nominal outcome. Save `execution_report.json`, event logs, and
   video for every acceptance run.

Run the matched comparison after data collection stops:

```bash
/isaac-sim/python.sh scripts/sim/run_rl_isaac_comparison.py \
  --run /workspace/team-data/collection_20261021_ready \
  --checkpoint models/rl_stacking_qtable.json \
  --vision-config configs/grasp_model_resnet_siamese.yaml \
  --seeds 310 311 312 313 314 315 316 317 318 319 \
          320 321 322 323 324 325 326 327 328 329
```

After classifier evaluation and the matched Isaac comparison, generate one
combined scorecard:

```bash
python scripts/generate_lab_readiness_report.py \
  --data /workspace/team-data/table-external-test-20261025-190359 \
  --classifier-metrics results/external_test/external_test_metrics.json \
  --rl-metrics results/rl_isaac_comparison/summary.json
```

This writes `results/lab_readiness/readiness.json` and a short Markdown summary.
The current collection target is 250 examples **per binary class**, or 500
independent test pairs total.

## Information to obtain from the lab before arrival

- Exact FANUC arm and controller model, supported control interface, and who
  is authorized to load/run programs.
- End-effector model, jaw limits, tool-center-point measurement, payload, and
  available digital I/O or robot handshake.
- Cube dimensions/material, table height, fixture layout, camera model/mount,
  network restrictions, and whether ROS/RMI/vendor drivers already exist.
- A known-safe home pose, joint/workspace limits, permitted speed, collision
  zones, and the lab's stop/recovery procedure.

If any of those are unknown, simulation cannot guarantee a direct hardware
run. Resolve them with the lab contact before the visit.

## Lab-day staged qualification

Do not begin with the six-cube tower.

1. **Safety briefing:** operator checks guarding, E-stop, reduced speed,
   payload, TCP, joint limits, home pose, and clear workspace.
2. **Shadow mode:** capture the scene and print planned poses without sending
   motion. Compare the reported cube centers and tower site against physical
   measurements.
3. **Dry motion:** with no cubes in the gripper, run approach/retreat paths at
   reduced speed and with extra vertical clearance. The operator validates
   every frame and waypoint.
4. **One cube:** complete five supervised pick/place cycles. Require 5/5 pickup
   verification and placement inside tolerance before proceeding.
5. **Two-cube mini-tower:** complete three consecutive towers without manual
   correction. Deliberately shift one cube once and confirm retry/safe-stop.
6. **Full tower:** run the frozen selected policy. Keep deterministic recovery
   as the fallback. Save logs and video; do not tune during the run.

## Go/no-go rule

Run the full demonstration only if the preceding stage passed. If calibration,
TCP, camera registration, gripper geometry, or verification disagrees with the
real cell, stop at that stage. A controlled two-cube demonstration with honest
metrics is stronger—and safer—than forcing an unverified six-cube motion.
