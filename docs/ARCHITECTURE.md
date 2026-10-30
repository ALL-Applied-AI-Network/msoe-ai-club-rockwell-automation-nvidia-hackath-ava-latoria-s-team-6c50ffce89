# Predictive Closed Loop Manipulation Architecture

Our system combines language-based task planning, learned visual verification,
deterministic robot control, bounded reinforcement learning, and continuous
telemetry. Learned components predict outcomes and recommend high-level
corrections; they never command robot joints or bypass the safety gates.

## System architecture

```mermaid
flowchart TB
  subgraph DEV["Development and training on NVIDIA Brev"]
    TWIN["NVIDIA Isaac Sim digital twin<br/>CSI FANUC ER-4iA workcell"]
    ROLLOUTS["Domain-randomized rollouts<br/>successes and intentional failures"]
    DATA["Before-and-after RGB pairs<br/>labels, seeds, and metadata"]
    TRAIN["PyTorch training and evaluation<br/>ResNet-18 Siamese model"]
    MODEL["Frozen checkpoint<br/>thresholds and evaluation metrics"]

    TWIN --> ROLLOUTS --> DATA --> TRAIN --> MODEL
  end

  subgraph PLAN["Task planning"]
    INSTRUCTION["Natural-language instruction"]
    GEMINI["Gemini instruction planner<br/>structured task output"]
    VALIDATE{"Schema and allowlist validation"}
    TASK["Validated task plan<br/>skill, source, destination, and size"]
    REJECT["Clarification or supervised safe stop"]

    INSTRUCTION --> GEMINI --> VALIDATE
    VALIDATE -->|Accepted| TASK
    VALIDATE -->|Rejected| REJECT
  end

  subgraph LOOP["Predictive closed-loop manipulation"]
    PERCEPTION["Table-camera perception<br/>cube color and position"]
    STATE["Recovery state machine<br/>locate through final verification"]
    FRAMES["Before-and-after RGB frames"]
    CLASSIFIER["Siamese grasp verifier<br/>prediction and confidence"]
    RL["Safety-shielded Q-learning<br/>bounded XY correction, retry, or stop"]
    MOTION["Inverse kinematics and<br/>interpolated trajectory"]
    SAFETY{"Workspace, tolerance,<br/>retry, and safe-stop gates"}
    EXECUTE["Execute permitted motion"]
    TARGET{"Execution target"}
    SIM["Isaac Sim<br/>digital-twin validation"]
    FANUC["CSI FANUC workcell<br/>supervised physical validation"]
    PLACE["Placement and tower verification"]

    TASK --> PERCEPTION --> STATE
    MODEL --> CLASSIFIER
    FRAMES --> CLASSIFIER -->|GOOD, BAD, or uncertain| STATE
    STATE --> MOTION
    STATE --> RL -->|Bounded correction| MOTION
    MOTION --> SAFETY --> EXECUTE --> TARGET
    TARGET --> SIM
    TARGET --> FANUC
    SIM --> FRAMES
    FANUC --> FRAMES
    FRAMES --> PLACE --> STATE
  end

  subgraph OBS["Observability and evidence"]
    API["Read-only Python dashboard API<br/>events, metrics, frames, and status"]
    DASH["React and TypeScript dashboard<br/>video synchronized with decisions"]
    REVIEW["Operator review<br/>performance evidence and failure analysis"]

    API --> DASH --> REVIEW
  end

  STATE -. state and retries .-> API
  CLASSIFIER -. prediction and confidence .-> API
  RL -. correction and action .-> API
  EXECUTE -. timing and motion status .-> API
  PLACE -. placement measurements .-> API

  classDef compute fill:#e8f2ff,stroke:#2563a6,color:#111827;
  classDef learned fill:#f0eaff,stroke:#6d4bc3,color:#111827;
  classDef control fill:#eaf7ed,stroke:#2f7d44,color:#111827;
  classDef safety fill:#fff4dd,stroke:#b7791f,color:#111827;
  classDef hardware fill:#fce8e8,stroke:#b23b3b,color:#111827;
  classDef telemetry fill:#eef2f6,stroke:#52606d,color:#111827;

  class TWIN,ROLLOUTS,DATA compute;
  class TRAIN,MODEL,GEMINI,CLASSIFIER,RL learned;
  class TASK,PERCEPTION,STATE,MOTION,PLACE control;
  class VALIDATE,REJECT,SAFETY safety;
  class EXECUTE,TARGET,SIM,FANUC,FRAMES hardware;
  class API,DASH,REVIEW telemetry;
```

## How the components work together

We used the provided NVIDIA Brev credits to run the Isaac Sim digital twin,
generate repeatable successes and intentional failures, and train the Siamese
grasp verifier on before-and-after RGB image pairs. The selected checkpoint is
then loaded by the runtime verifier.

At runtime, Gemini converts the language instruction into a structured task
plan. A schema and allowlist gate rejects ambiguous or unsupported plans before
motion begins. Table-camera perception locates the source and destination
objects, and the deterministic recovery state machine sequences the manipulation
steps. The Siamese model predicts whether the grasp succeeded, while the
safety-shielded Q-learning policy can propose only a small placement correction,
retry, or stop. Inverse kinematics, trajectory interpolation, workspace limits,
position tolerances, and retry limits remain authoritative.

The same control path can target the Isaac Sim workcell or the supervised CSI
FANUC cell. Every prediction, state transition, RL decision, timing measurement,
and verification result is sent one way to the read-only dashboard API. The
React and TypeScript dashboard synchronizes those records with video so an
operator can see what the system predicted, what it decided, and whether the
action succeeded.

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
