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

## Read-only dashboard API

Browsers cannot safely poll a file inside the Brev workspace directly. Start
the dependency-free API bridge against the exact output folder being shown:

```bash
python scripts/serve_dashboard_api.py \
  --run /workspace/team-data/e2e/sort-stack-fast-01 \
  --host 0.0.0.0 \
  --port 8765 \
  --allow-origin http://localhost:5173
```

Use `127.0.0.1` instead of `0.0.0.0` when the frontend runs on the same machine.
On Brev, expose port `8765` through an authenticated HTTP port and add that
dashboard's exact `https://...gobrev.dev` origin with another `--allow-origin`.
Do not use `*` for a public dashboard.

Because Brev's authenticated HTTP port uses a Pomerium session cookie, browser
requests from the React app must opt in to credentials:

```ts
const response = await fetch(`${apiBase}/api/status`, {
  credentials: "include",
});
```

The API returns credentialed CORS headers only for the exact origins supplied
with `--allow-origin`; it never combines credentials with a wildcard origin.

Endpoints:

- `GET /health`
- `GET /api/dashboard` — one frontend-ready object with display cards, metrics,
  progress, media URLs and links to detailed data
- `GET /api/status`
- `GET /api/events?stage=all|sort|stack`
- `GET /api/metrics`
- `GET /api/artifacts`
- `GET /api/video` — automatically serves the preferred MP4/WebM in the run
  folder and supports browser byte-range requests
- `GET /api/latest-frame?stage=sort|stack&camera=main|side`
- `GET /artifacts/<run-relative-path>`
- `GET /media/<run-relative-video-or-image>`

The bridge is read-only, blocks path traversal, supports video byte ranges, and
never starts, stops, or commands the robot. JSON responses use `no-store` so a
frontend can poll without seeing stale status.

For Ana's dashboard, poll `/api/dashboard` instead of assembling the cards from
several files. The response's `display` object maps directly to the current UI:
`phase`, `prediction`, `confidence`, `state_machine_step`, `rl_correction`,
`selected_action`, `grasp_result`, `cycle_time_s`, `placement_error_mm`, and
`overall_success_rate`. Set the video source to
`${apiBase}${dashboard.media.video_url}` when `video_url` is not null. The API
discovers a recording after it is placed in the run folder; no filename needs
to be hard-coded. The `timeline` array contains the real completed grasp events
with cumulative `start_s` and `end_s` values, so the frontend can update the
side panel as the recorded video's `currentTime` crosses an event boundary.
`timeline_basis` states that these boundaries come from recorded cycle times,
not hand-authored demo values.

Completed runs that predate `live_status.json` are also supported. The API reads
the real `grabs` array from `end_to_end_metrics.json`, preserves missing
classifier scores as `null`, and constructs the timeline from each recorded
`cycle_time_s`. The React player scales that cumulative timeline to the loaded
recording's actual duration so seeking remains synchronized even if video
encoding slightly changes its duration.

The response also has a top-level `planning` object. It contains only the
validated, allow-listed task plan that is safe to show: planner source
(`deterministic` or `gemini`), instruction, action, source and destination,
object shape and size, a readable summary, and a `ready`, `waiting`, or
`blocked` status. It never includes the Gemini key, prompt, or raw model
response. The LLM may interpret an instruction, but it does not generate robot
motion; the state machine and safety checks remain authoritative.

The browser still needs a browser-compatible recording. Prefer H.264 in an MP4
container or WebM; an Isaac/OpenCV `mp4v` recording may need to be transcoded
before Chrome or Edge can play it.

The easiest hackathon frontend is React + TypeScript + Vite. Use the exported
types in `liveStatus.ts`. Recorded MP4 playback can be embedded directly. The
authenticated Brev WebRTC viewer should initially be an **Open live twin** link
rather than an iframe.

## Run Ana's dashboard

In one terminal, start the read-only API for the run folder you want to show:

```bash
python scripts/serve_dashboard_api.py --run /workspace/team-data/e2e/sort-stack-visual-02
```

In another terminal:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Vite sends `/api`, `/media`, and `/artifacts` to the local API on port 8765.
For a separately hosted API, set `VITE_DASHBOARD_API_BASE` to its authenticated
base URL before starting Vite. Do not put `GEMINI_API_KEY` or any other secret
in a `VITE_` variable because Vite variables are visible in the browser.

Fast simulator run:

```bash
/isaac-sim/python.sh scripts/sim/run_sort_then_stack.py \
  --run /workspace/team-data/grasp-sort-live/sort_baseline_full_02 \
  --out /workspace/team-data/e2e/sort-stack-fast-01 \
  --fast
```

`--fast` changes only digital-twin interpolation speed. It does not configure
or authorize a real FANUC speed override.

Final integrated demo (rendered classifier gate + bounded state-machine
recovery + RL placement correction):

```bash
/isaac-sim/python.sh scripts/sim/run_sort_then_stack.py \
  --run /workspace/team-data/grasp-sort-live/sort_baseline_full_02 \
  --out /workspace/team-data/e2e/sort-stack-integrated-01 \
  --integrated-demo
```

In live status, `current.rl.action` is the latest action for a simple status
card and `current.rl.actions` is the complete bounded policy trace.
