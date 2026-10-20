# Vision-Guided Closed-Loop Manipulation for Robust Cube Stacking

## Problem

The FANUC starter detects and plans grasps with the table camera, then executes **open loop**. The wrist camera is unused during pick/place, so perception success does not guarantee manipulation success.

## Method

- **`grasp_check` / `seat_check`** on wrist image after jaw close
- **`place_check`** after release
- **Recovery policy:** up to 2 grasp retries, 1 placement retry, then safe stop with `FailureReason` codes
- **JSONL** trace: `manipulation.jsonl` per run
- **Benchmark:** open loop vs closed loop × stock vs cube finger (`configs/trials/`)

## Results

Copy only from [`results/benchmarks/summary.md`](../../results/benchmarks/summary.md).

| Metric | Baseline (open loop) | Closed loop |
| --- | --- | --- |
| First-grasp success | | |
| Tower completion | | |
| Recovery success | | |
| Mean retries | | |
| Mean cycle time (s) | | |

## Remaining failures

Document `TILTED_OBJECT`, `OCCLUSION`, and max-retry stops honestly in the benchmark table.

## Hardware vs sim

State what ran on the supervised FANUC arm vs Isaac-only cells.

## Credits

FANUC ER-4iA hackathon starter, Isaac Sim, Cosmos3-Nano, team repository.
