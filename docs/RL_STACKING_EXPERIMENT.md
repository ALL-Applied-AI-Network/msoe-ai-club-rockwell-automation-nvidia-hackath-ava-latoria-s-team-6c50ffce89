# RL Stacking Experiment (`ava-rl`)

This branch explores reinforcement learning as an **extra simulation feature**.
It does not replace the normal stacking pipeline or send commands directly to
the physical FANUC arm.

## What the agent learns

For each cube, the policy sees a small state:

- current stack height;
- whether the cube is left/right or forward/back of the target;
- grasp quality from the verifier;
- stack stability; and
- retries already used.

It chooses one bounded action: place in the center, apply one of four small
placement corrections, retry the grasp, or stop safely. Successful placements
and completed stacks earn rewards. Drops, collapsed stacks, unnecessary retries,
and unsafe choices receive penalties.

This is high-level RL rather than direct robot-joint control. That makes it
fast enough to train during the hackathon, easy to explain, and compatible with
the existing recovery state machine. Isaac Sim can later replace the lightweight
training environment while keeping the same state/action contract.

## Run it

From the repository root:

```bash
python scripts/train_rl_stacking.py
```

The default run trains for 50,000 short simulated episodes and then evaluates
on 1,000 unseen seeds. It writes:

- `models/rl_stacking_qtable.json` — learned policy/checkpoint;
- `results/rl_stacking_metrics.json` — success rate and reward summary.

The metrics use the same hidden evaluation seeds for the learned policy and an
open-loop baseline that retries empty grasps but does not correct placement.

For a quick check:

```bash
python scripts/train_rl_stacking.py --episodes 2000 --eval-episodes 200
```

To continue an existing checkpoint without replacing it:

```bash
python scripts/train_rl_stacking.py \
  --resume models/rl_stacking_qtable.json \
  --episodes 200000 --eval-episodes 10000 \
  --out models/rl_stacking_candidate.json \
  --metrics results/rl_stacking_candidate_metrics.json
```

Always compare candidates on the same evaluation seeds before selecting one.
The current selected checkpoint remains `rl_stacking_qtable.json`: it scored
64.55% on 10,000 fixed unseen episodes, while the 200,000-episode continuation
scored 53.61% and was rejected. More episodes alone can make a tabular policy
worse when its learning rate remains constant.

## How it connects to Isaac Sim

`IsaacRLPlacementAdapter` converts a hidden placement offset into
`StackingState`, asks the safety-shielded policy for bounded corrections, and
passes only the corrected XY destination to the existing IK/trajectory code.
The existing code still owns joint motion, workspace limits, grasp checks, and
placement verification.

Prepare five matched seeds without starting Isaac:

```bash
python scripts/sim/run_rl_isaac_comparison.py \
  --run /workspace/team-data/collection_20261021_ready \
  --prepare-only
```

After the current data collector finishes, run the actual comparison:

```bash
/isaac-sim/python.sh scripts/sim/run_rl_isaac_comparison.py \
  --run /workspace/team-data/collection_20261021_ready \
  --checkpoint models/rl_stacking_qtable.json \
  --vision-config configs/grasp_model_resnet_siamese.yaml
```

Do not run this beside another Isaac collector on the L4. The script launches
one simulation process and evaluates open-loop, deterministic closed-loop, and
RL on identical seeds and hidden placement offsets. Results go to
`results/rl_isaac_comparison/summary.json`.

The RL path is visually guarded. After lift, the Siamese classifier checks the
table-camera before/after pair. A high-confidence `GOOD` may proceed; a
borderline result gets one second look; a predicted failure, invalid crop, or
two-look disagreement stops before the carry-to-stack segment. The classifier
does not command joints—RL only adjusts the XY destination, and the existing
IK, retry limits, and placement checks remain in control.

For the demo, compare the same hidden scene seeds across:

1. open-loop stacking;
2. the deterministic recovery state machine; and
3. the safety-shielded RL policy.

Report stack-completion rate, cube drops, safe stops, retries, and mean time per
successful stack. Only move a frozen, safety-shielded policy toward the real
FANUC after supervised review and workspace limits are confirmed.
