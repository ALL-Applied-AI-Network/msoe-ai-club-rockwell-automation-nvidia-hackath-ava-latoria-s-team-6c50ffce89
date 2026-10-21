#!/usr/bin/env bash
set -euo pipefail

# Run this from the Brev Jupyter terminal. It creates the shared USD stage used
# by the browser viewer without changing any source files.
REPO_DIR="${TEAM_REPO_DIR:-/workspace/team-repo}"
SIM_DIR="${REPO_DIR}/data/fanuc_er4ia_hackathon_starter_v05 (1)/fanuc_er4ia_sim"
DATA_DIR="${TEAM_DATA_DIR:-/workspace/team-data}"
STAGE="${DATA_DIR}/fanuc_cell.usd"
ISAAC_DIR="${ISAAC:-/isaac-sim}"

if [[ ! -x "${ISAAC_DIR}/python.sh" ]]; then
  echo "Isaac Sim was not found at ${ISAAC_DIR}."
  exit 1
fi
if [[ ! -f "${SIM_DIR}/build_scene.py" ]]; then
  echo "The FANUC starter was not found at ${SIM_DIR}."
  exit 1
fi
if pgrep -x kit >/dev/null 2>&1; then
  echo "Isaac Sim is already running. Stop the viewer before rebuilding the stage."
  exit 1
fi

mkdir -p "${DATA_DIR}"
cd "${SIM_DIR}"
echo "Building the FANUC workcell. The first build can take several minutes..."
"${ISAAC_DIR}/python.sh" build_scene.py --headless --save "${STAGE}"

test -s "${STAGE}"
echo "Digital twin stage is ready: ${STAGE}"
