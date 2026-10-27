# Vision-Guided Closed-Loop Manipulation for Robust Cube Stacking

**AI design (target):** [AI_GRASP_VERIFICATION_DESIGN.md](AI_GRASP_VERIFICATION_DESIGN.md) — synthetic wrist dataset, multi-class grasp-quality classifier, ablation vs rules vs open loop.

## Problem

The FANUC starter detects cubes with the **table camera**, plans grasps, and executes **open loop**. The **wrist camera** is calibrated but unused during pick/place.

## Approach

After jaw close → wrist **grasp quality** (`rules` today → **learned classifier** per design doc). After release → place verification. On failure, bounded regrasp with class-specific recovery. Benchmark **baseline / rules / AI** on identical seeds.

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/stacking_recovery/`](../src/stacking_recovery/) | Library: vision, recovery policy, metrics |
| [`configs/trials/`](../configs/trials/) | Perturbation + policy matrix |
| [`configs/finger/`](../configs/finger/) | Finger geometry params |
| [`scripts/`](../scripts/) | `run_benchmark.py`, `generate_trials.py`, `aggregate_results.py` |
| [`tests/`](../tests/) | Unit tests (no Isaac) |
| [`data/fixtures/`](../data/fixtures/) | Lab/sim wrist crops |
| [`results/`](../results/) | Sim runs and benchmark summary |
| [`data/.../fanuc_er4ia_sim/`](../data/fanuc_er4ia_hackathon_starter_v05%20(1)/fanuc_er4ia_sim/) | Vendor sim (`sort_scene.py` integration) |
| [`stretch/yolo/`](../stretch/yolo/) | Optional detector swap (after core ships) |

## Sim integration

- [`sort_scene.py`](../data/fanuc_er4ia_hackathon_starter_v05%20(1)/fanuc_er4ia_sim/sort_scene.py): `--closed-loop`, `--trial-config`
- [`run_sort_pipeline.sh`](../data/fanuc_er4ia_hackathon_starter_v05%20(1)/fanuc_er4ia_sim/run_sort_pipeline.sh): `CLOSED_LOOP=1`, `TRIAL_CONFIG=...`
- Logs: `manipulation.jsonl` in the run directory

## Contracts

- `grasp_check(image_bgr) -> GraspCheckResult` (`seat_check` alias)
- `place_check(image_bgr, stack_level) -> PlaceCheckResult`
- Schemas: [`configs/schemas/`](../configs/schemas/)
