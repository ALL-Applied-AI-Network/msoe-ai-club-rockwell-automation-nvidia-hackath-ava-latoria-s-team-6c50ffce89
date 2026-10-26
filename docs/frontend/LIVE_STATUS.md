# React dashboard data

The simulator writes `live_status.json` inside every sort-then-stack output
folder. The file is replaced atomically, so a React/TypeScript frontend can poll
it without reading half-written JSON.

During a run it contains:

- overall state, phase and progress;
- the latest completed grasp;
- pickup, placement and classifier metrics;
- classifier prediction, confidence and camera-quality reason;
- bounded RL action, safe-stop state and retry counts;
- every completed event and links to final report artifacts.

Poll the file every 500–1000 ms. Treat `state` as the top-level UI state and
`current` as the card for the latest grasp. Show `NO SCORE` when
`current.camera_quality.valid` is false; do not display that as a good grasp.

The easiest hackathon frontend is React + TypeScript + Vite. Use the exported
types in `liveStatus.ts`. Recorded MP4 playback can be embedded directly. The
authenticated Brev WebRTC viewer should initially be an **Open live twin** link
rather than an iframe.

Fast simulator run:

```bash
/isaac-sim/python.sh scripts/sim/run_sort_then_stack.py \
  --run /workspace/team-data/grasp-sort-live/sort_baseline_full_02 \
  --out /workspace/team-data/e2e/sort-stack-fast-01 \
  --fast
```

`--fast` changes only digital-twin interpolation speed. It does not configure
or authorize a real FANUC speed override.
