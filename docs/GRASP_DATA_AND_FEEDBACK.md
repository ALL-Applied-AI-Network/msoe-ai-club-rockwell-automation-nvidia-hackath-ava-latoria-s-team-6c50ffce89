# Grasp dataset collection and continual retraining

## Modes

| Mode | Flag / config | When samples are saved |
|------|----------------|-------------------------|
| **Off** | default | Never |
| **Feedback** | `--grasp-feedback` or `feedback.enabled: true`, `mode: feedback` | Bad grasp (verifier says not seated), physics mismatch after lift, safe-stop before lift |
| **Collect all** | `--collect-dataset PATH` | Every pick: wrist at dwell, weak label from physics at lift (Isaac batch) |

Labels at **lift** use physics only (`pickup_verified`, yaw, jaw gap) — never sim GT poses at runtime.

## Isaac (Brev) batch collection

```bash
cd data/fanuc_er4ia_hackathon_starter_v05\ \(1\)/fanuc_er4ia_sim
# After capture + plan for a run directory:
$ISAAC/python.sh sort_scene.py --run "$RUN" --collect-dataset /path/to/repo/data/grasp_quality \
  --trial-config /path/to/configs/trials/pick_xy_shift_closed_loop_seed0.json
```

Or use the wrapper (loops trials):

```bash
$ISAAC/python.sh scripts/sim/collect_wrist_dataset.py --repo /path/to/repo --runs-dir ...
```

Images append to `data/grasp_quality/manifest.json` with `"source": "isaac"`.

## Closed loop + live improvement

Sim or lab with AI/rules verification:

```bash
CLOSED_LOOP=1 VERIFICATION=ai GRASP_FEEDBACK=1 ./run_sort_pipeline.sh "$RUN"
```

Or in `sort_scene.py`: `--closed-loop --verification ai --grasp-feedback`

After a session:

```bash
python scripts/retrain_grasp_classifier.py --finetune
```

This loads the current `.pt`, trains on **full manifest** (procedural + isaac + feedback) with augmentations and a lower LR.

## Manifest layout

```
data/grasp_quality/
  manifest.json
  feedback_log.jsonl
  images/train/{LABEL}/{source}_{id}.png
```

## Hooks in `sort_scene.py`

1. **dwell** — wrist PNG + `GraspFeedbackRecorder.on_dwell`
2. **lift** — `pickup_verified` → `on_lift(event)` writes labeled sample
3. **grasp safe-stop** — `on_grasp_failure_no_lift` when no lift occurs

## Team workflow

1. Procedural baseline: `generate_procedural_wrist_dataset.py` → `train_grasp_classifier.py`
2. Brev: `collect_all` rollouts across trial seeds
3. Deploy `VERIFICATION=ai` with `GRASP_FEEDBACK=1`
4. Weekly (or after lab): `retrain_grasp_classifier.py --finetune`
5. Spot-check `feedback_log.jsonl` for label noise before trusting fine-tune
