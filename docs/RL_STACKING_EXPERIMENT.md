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

## How it connects to Isaac Sim

The next step is an adapter that converts each Isaac result into
`StackingState`, executes the chosen bounded placement/recovery action through
the existing motion planner, and gives the agent a reward after the placement
verification step. We should train and evaluate that adapter only in Isaac Sim.

For a strong demo, compare the same hidden scene seeds across:

1. open-loop stacking;
2. the deterministic recovery state machine; and
3. the safety-shielded RL policy.

Report stack-completion rate, cube drops, safe stops, retries, and mean time per
successful stack. Only move a frozen, safety-shielded policy toward the real
FANUC after supervised review and workspace limits are confirmed.
