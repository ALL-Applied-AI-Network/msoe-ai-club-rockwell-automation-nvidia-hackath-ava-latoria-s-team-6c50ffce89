# Contributing

Work by **component**, not by person. Multiple teammates may touch the same module; use branches and small PRs.

## Where to code

| Component | Location |
| --- | --- |
| Wrist grasp verification (rules) | `src/stacking_recovery/vision/grasp_check.py` |
| Wrist grasp verification (AI) | `src/stacking_recovery/vision/grasp_quality_model.py`, `grasp_check_factory.py` |
| Procedural / Isaac datasets | `src/stacking_recovery/data/`, `scripts/generate_procedural_wrist_dataset.py`, `scripts/sim/collect_wrist_dataset.py` |
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

## Closed-loop sim run

From the FANUC starter directory (with Isaac/Cosmos configured):

```bash
CLOSED_LOOP=1 VERIFICATION=ai TRIAL_CONFIG=/path/to/configs/trials/nominal_closed_loop_seed0.json \
  ./run_sort_pipeline.sh /path/to/results/runs/my_trial
```

Train the classifier locally before Brev: `pip install -e ".[ml]"` then see [models/README.md](../models/README.md).

Generate the full trial matrix: `python scripts/generate_trials.py`

## Rules

- Do not commit Isaac runs, videos, or model weights (see `.gitignore`).
- Benchmark must report failures honestly (e.g. tilted cube).
- Closed-loop retries: max **2** grasp, **1** placement retry unless integrator changes constants in `recovery_policy.py`.
