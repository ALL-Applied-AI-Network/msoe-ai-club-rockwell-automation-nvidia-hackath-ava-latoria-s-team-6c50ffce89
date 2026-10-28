# Robot demonstration dashboard

This is Ana's React/TypeScript dashboard connected to the project's read-only
dashboard API. It shows the run video beside the classifier, state-machine, RL,
placement and language-planning information that was recorded for that run.

## Start it

First start the API from the repository root:

```bash
python scripts/serve_dashboard_api.py --run /path/to/the/run-folder
```

Then open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite. The development server automatically forwards
the API and video requests to `127.0.0.1:8765`.

The run folder should contain the project's JSON output and a browser-compatible
H.264 MP4 or WebM recording. The dashboard discovers the recording through the
backend; no video filename is hard-coded.

If the API is hosted at a different origin, set `VITE_DASHBOARD_API_BASE` to its
base URL. Never put `GEMINI_API_KEY` in a `VITE_` variable—the browser must not
receive the key.
