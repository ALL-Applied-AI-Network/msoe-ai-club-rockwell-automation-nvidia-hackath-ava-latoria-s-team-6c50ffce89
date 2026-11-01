# Manipulation dashboard

This React/TypeScript app pairs the simulator or remote FANUC recording with
the model evidence that produced each decision. It never sends robot commands.

The app has five operations views:

- **Run monitor** — synchronized video, classifier result, state-machine step,
  safety gates, task plan and event timeline;
- **Metrics** — measured classifier, confusion-matrix and bounded-RL results;
- **Grasp review** and **Stack validation** — human GOOD/BAD labels queued as
  evidence for a later, supervised offline retraining pass;
- **Instruction studio** — a dry-run task-plan preview using the existing
  deterministic/Gemini planning boundary;
- **Videos & runs** — the full digital-twin recording plus all 12 table-camera
  episodes from the latest CSI trial, with side and wrist views where available.

## Run locally

From the repository root, start the API with either a simulator run folder or
a remote telemetry session folder:

```bash
python scripts/serve_dashboard_api.py --run /path/to/run --allow-origin http://localhost:5173
```

Then start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies API requests to port 8765. To use a
different backend, set `VITE_DASHBOARD_API_BASE` in `frontend/.env.local`.

When the API is offline, the checked-in Isaac Sim video still plays, but the UI
labels it as an unsynchronized reference and does not invent model values.

Human labels are appended to `dashboard_feedback.jsonl` in the selected run
folder. They do not retrain or deploy a model automatically. Before retraining,
the labels must be paired with their source before/after frames, reviewed, and
kept out of the held-out test set.
