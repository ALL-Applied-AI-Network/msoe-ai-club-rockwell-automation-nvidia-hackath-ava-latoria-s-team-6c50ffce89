# Contributing

Work by **component**, not by person. Multiple teammates may touch the same module; use branches and small PRs.

## Where to code

| Component | Location |
| --- | --- |
| Wrist grasp verification | `src/stacking_recovery/vision/grasp_check.py` |
| Placement verification | `src/stacking_recovery/vision/place_check.py` |
| Recovery state machine | `src/stacking_recovery/control/recovery_policy.py` |
| Isaac / `sort_scene` wiring | `src/stacking_recovery/integration/sim_hooks.py` |
| Perturbation loading | `src/stacking_recovery/bench/perturbations.py` |
| Metric aggregation | `src/stacking_recovery/bench/metrics.py` |
| Benchmark driver | `scripts/run_benchmark.py` |
| Trial definitions | `configs/trials/*.json` |
| Finger geometry params | `configs/finger/*.json` → sim via integration |
| Submission copy | `docs/submission/` |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
python -m unittest discover -s tests -p "test_*.py"
```

## Rules

- Do not commit Isaac runs, videos, or model weights (see `.gitignore`).
- Benchmark must report failures honestly (e.g. tilted cube).
- Closed-loop retries: max **2** grasp, **1** placement retry unless integrator changes constants in `recovery_policy.py`.
