#!/usr/bin/env bash
set -euo pipefail

# Run this inside the Brev Isaac/Jupyter container. It starts the browser
# stream with the saved workcell and the safe looping motion demonstration.
REPO_DIR="${TEAM_REPO_DIR:-/workspace/team-repo}"
DATA_DIR="${TEAM_DATA_DIR:-/workspace/team-data}"
ISAAC_DIR="${ISAAC:-/isaac-sim}"
VIEWER_SCRIPT="${REPO_DIR}/scripts/brev/open_fanuc_viewer.py"
STAGE="${DATA_DIR}/fanuc_cell.usd"
LOG="${DATA_DIR}/fanuc-stream.log"

if [[ ! -x "${ISAAC_DIR}/runheadless.sh" ]]; then
  echo "Isaac Sim was not found at ${ISAAC_DIR}."
  exit 1
fi
if [[ ! -f "${VIEWER_SCRIPT}" ]]; then
  echo "Viewer helper not found: ${VIEWER_SCRIPT}"
  exit 1
fi
if [[ ! -s "${STAGE}" ]]; then
  echo "Digital twin stage not found: ${STAGE}"
  echo "Run: bash scripts/brev/build_fanuc_stage.sh"
  exit 1
fi

if pgrep -x kit >/dev/null 2>&1; then
  echo "Stopping the existing Isaac Sim session..."
  pkill -TERM -x kit
  for _ in $(seq 1 30); do
    pgrep -x kit >/dev/null 2>&1 || break
    sleep 1
  done
fi

mkdir -p "${DATA_DIR}"
rm -f "${LOG}"
nohup env ACCEPT_EULA=Y "${ISAAC_DIR}/runheadless.sh" \
  --exec "${VIEWER_SCRIPT}" >"${LOG}" 2>&1 &

echo "Isaac Sim is starting in the background."
echo "Watch startup: tail -f ${LOG}"
echo "It is ready when the log says: safe looping demonstration started"
