# Vision-Guided Closed-Loop Manipulation for Robust Cube Stacking

## Problem

The FANUC starter detects cubes with the **table camera**, plans grasps, and executes **open loop**. The **wrist camera** is calibrated but unused during pick/place. Perception success does not guarantee manipulation success.

## Approach

After jaw close → `grasp_check(wrist_image)`. After release → `place_check(image)`. On failure, bounded regrasp with explicit `FailureReason` codes. Compare **open loop** vs **closed loop** on identical seeds, optionally × **stock finger** vs **cube pad**.

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/stacking_recovery/`](../src/stacking_recovery/) | Library: vision checks, recovery policy, metrics |
| [`configs/`](../configs/) | Trial JSON, finger params, benchmark settings |
| [`scripts/`](../scripts/) | CLI entry points (benchmark, aggregate, overlay) |
| [`tests/`](../tests/) | Unit tests (no Isaac) |
| [`data/fixtures/`](../data/fixtures/) | Small labeled wrist/table crops |
| [`results/`](../results/) | Benchmark outputs and sim run dirs (gitignored) |
| [`hardware/`](../hardware/) | CAD exports for printed finger |
| [`docs/submission/`](../docs/submission/) | One-pager, video script, lab notes |
| [`external/fanuc_er4ia_sim/`](../data/fanuc_er4ia_hackathon_starter_v05%20(1)/fanuc_er4ia_sim/) | Vendor starter (integrate via `integration/sim_hooks.py`) |
| [`stretch/yolo/`](../stretch/yolo/) | Optional detector swap (after core ships) |

## Integration boundary

Only [`src/stacking_recovery/integration/sim_hooks.py`](../src/stacking_recovery/integration/sim_hooks.py) and the starter’s `sort_scene.py` connect to Isaac. Do not weaken perception gates in `plan_sort.py` to force passes.

## Contracts

- `grasp_check(image_bgr) -> GraspCheckResult` — seated, yaw_error_deg, confidence, reason  
- `place_check(image_bgr, stack_level) -> PlaceCheckResult` — placed, confidence, reason  
- Events: JSONL via `stacking_recovery.events`  
- Trials: JSON under `configs/trials/` matching `configs/schemas/perturbation.schema.json`
