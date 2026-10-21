#!/usr/bin/env bash
set -euo pipefail

# Persistent Brev storage survives an instance stop. Teammates on this instance
# can source the generated file and use the exact same dataset location. The
# Isaac/Jupyter container uses /workspace; the Brev host uses /home/ubuntu/workspace.
if [[ -d /isaac-sim && -d /workspace ]]; then
  DEFAULT_BREV_WORKSPACE=/workspace
else
  DEFAULT_BREV_WORKSPACE=/home/ubuntu/workspace
fi
BREV_WORKSPACE="${BREV_WORKSPACE:-${DEFAULT_BREV_WORKSPACE}}"
TEAM_REPO_DIR="${TEAM_REPO_DIR:-${BREV_WORKSPACE}/team-repo}"
GRASP_DATA_ROOT="${GRASP_DATA_ROOT:-${BREV_WORKSPACE}/team-data/grasp_quality}"
ENV_FILE="${BREV_WORKSPACE}/team-data.env"

mkdir -p "${GRASP_DATA_ROOT}"
printf 'export TEAM_REPO_DIR=%q\n' "${TEAM_REPO_DIR}" > "${ENV_FILE}"
printf 'export GRASP_DATA_ROOT=%q\n' "${GRASP_DATA_ROOT}" >> "${ENV_FILE}"
printf 'export ISAAC=%q\n' "${ISAAC:-/isaac-sim}" >> "${ENV_FILE}"

echo "Shared dataset workspace is ready."
echo "Run: source ${ENV_FILE}"
echo "Dataset: ${GRASP_DATA_ROOT}"
