#!/usr/bin/env bash
set -euo pipefail

# Beginner-friendly background collector for the fixed table camera.
# Run inside the Brev Jupyter terminal (the vscode container).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TEAM_REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ISAAC_DIR="${ISAAC:-/isaac-sim}"
RUN_DIR="${TEAM_RUN_DIR:-/workspace/team-data/collection_20261021_ready}"
JOB_DIR="${TEAM_DATA_DIR:-/workspace/team-data}/table-collection-job"
PID_FILE="${JOB_DIR}/collector.pid"
LOG_FILE="${JOB_DIR}/collector.log"
LATEST_FILE="${JOB_DIR}/latest-dataset.txt"

running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

worker() {
  local dataset="$1"
  local picks="$2"
  local sim="${REPO_DIR}/data/fanuc_er4ia_hackathon_starter_v05 (1)/fanuc_er4ia_sim/sort_scene.py"
  local trials=(
    "nominal_open_loop_seed0.json"
    "yaw_error_rotation_open_loop_seed2.json"
    "pick_xy_shift_validation_open_loop_seed1.json"
    "yaw_error_rotation_open_loop_seed1.json"
    "pick_xy_shift_severe_open_loop_seed5.json"
  )

  for trial in "${trials[@]}"; do
    local name="${trial%.json}"
    echo "[$(date -Is)] collecting ${name}"
    "${ISAAC_DIR}/python.sh" "${sim}" \
      --run "${RUN_DIR}" \
      --collect-dataset "${dataset}" \
      --trial-config "${REPO_DIR}/configs/trials/${trial}" \
      --out "${dataset}/run-${name}" \
      --max-picks "${picks}"
  done

  echo "[$(date -Is)] validating dataset"
  PYTHONPATH="${REPO_DIR}/src" "${ISAAC_DIR}/python.sh" \
    "${REPO_DIR}/scripts/validate_grasp_dataset.py" --data "${dataset}"
  echo "[$(date -Is)] COLLECTION COMPLETE: ${dataset}"
  echo "Stop the Brev environment now so it does not keep using credits."
}

balance_worker() {
  local dataset="$1"
  local sim="${REPO_DIR}/data/fanuc_er4ia_hackathon_starter_v05 (1)/fanuc_er4ia_sim/sort_scene.py"
  local shifted="${REPO_DIR}/configs/trials/pick_xy_shift_severe_open_loop_seed2.json"
  local nominal="${REPO_DIR}/configs/trials/nominal_open_loop_seed5.json"

  # A failed pickup ends a stack attempt, so launch each planned cube as its
  # own trial. This produces useful train failures instead of stopping after one.
  for pick_index in 0 1 2 3 4 5; do
    echo "[$(date -Is)] collecting train failure for pick ${pick_index}"
    "${ISAAC_DIR}/python.sh" "${sim}" \
      --run "${RUN_DIR}" \
      --collect-dataset "${dataset}" \
      --trial-config "${shifted}" \
      --pick-index "${pick_index}" \
      --out "${dataset}/run-shift-train-${pick_index}"
  done

  # The original dataset had no successful test examples. A nominal run fills
  # that gap while leaving the held-out test split untouched by training.
  echo "[$(date -Is)] collecting nominal test successes"
  "${ISAAC_DIR}/python.sh" "${sim}" \
    --run "${RUN_DIR}" \
    --collect-dataset "${dataset}" \
    --trial-config "${nominal}" \
    --out "${dataset}/run-nominal-test" \
    --max-picks 6

  echo "[$(date -Is)] validating balanced dataset"
  PYTHONPATH="${REPO_DIR}/src" "${ISAAC_DIR}/python.sh" \
    "${REPO_DIR}/scripts/validate_grasp_dataset.py" --data "${dataset}"
  echo "[$(date -Is)] BALANCE COLLECTION COMPLETE: ${dataset}"
  echo "Next: bash scripts/brev/train_table_pickup_model.sh"
}

campaign_worker() {
  local dataset="$1"
  local train_target="$2"
  local val_target="$3"
  local test_target="$4"
  "${ISAAC_DIR}/python.sh" "${REPO_DIR}/scripts/sim/collect_balanced_table_dataset.py" \
    --repo "${REPO_DIR}" \
    --run "${RUN_DIR}" \
    --out "${dataset}" \
    --isaac "${ISAAC_DIR}" \
    --target-train "${train_target}" \
    --target-val "${val_target}" \
    --target-test "${test_target}"

  echo "[$(date -Is)] validating balanced campaign dataset"
  PYTHONPATH="${REPO_DIR}/src" "${ISAAC_DIR}/python.sh" \
    "${REPO_DIR}/scripts/validate_grasp_dataset.py" --data "${dataset}"
  echo "[$(date -Is)] CAMPAIGN COLLECTION COMPLETE: ${dataset}"
  echo "Next: bash scripts/brev/train_table_pickup_model.sh"
}

command="${1:-help}"
case "${command}" in
  start)
    if running; then
      echo "A collector is already running with PID $(cat "${PID_FILE}")."
      exit 1
    fi
    mode="${2:-quick}"
    case "${mode}" in
      quick) picks=2 ;;
      full) picks=6 ;;
      *) echo "Use: $0 start [quick|full]"; exit 2 ;;
    esac
    if pgrep -x kit >/dev/null 2>&1; then
      echo "Isaac Sim is already running. Stop the viewer before collecting."
      echo "Run: bash scripts/brev/stop_fanuc_viewer.sh"
      exit 1
    fi
    [[ -x "${ISAAC_DIR}/python.sh" ]] || { echo "Isaac Sim not found at ${ISAAC_DIR}"; exit 1; }
    [[ -f "${RUN_DIR}/sort_plan.json" ]] || { echo "Prepared run not found: ${RUN_DIR}"; exit 1; }
    mkdir -p "${JOB_DIR}"
    dataset="${TEAM_DATA_DIR:-/workspace/team-data}/table-grasp-$(date +%Y%m%d-%H%M%S)"
    printf '%s\n' "${dataset}" >"${LATEST_FILE}"
    nohup bash "$0" _worker "${dataset}" "${picks}" >"${LOG_FILE}" 2>&1 &
    printf '%s\n' "$!" >"${PID_FILE}"
    echo "Collection started in the background (PID $!)."
    echo "Dataset: ${dataset}"
    echo "Watch it: bash scripts/brev/collect_table_dataset.sh watch"
    echo "When complete, stop the Brev environment from the Brev dashboard."
    ;;
  balance)
    if running; then
      echo "A collector is already running with PID $(cat "${PID_FILE}")."
      exit 1
    fi
    if pgrep -x kit >/dev/null 2>&1; then
      echo "Isaac Sim is already running. Stop the viewer before collecting."
      echo "Run: bash scripts/brev/stop_fanuc_viewer.sh"
      exit 1
    fi
    [[ -f "${LATEST_FILE}" ]] || { echo "No prior dataset is recorded."; exit 1; }
    dataset="$(cat "${LATEST_FILE}")"
    [[ -f "${dataset}/manifest.json" ]] || { echo "Dataset not found: ${dataset}"; exit 1; }
    nohup bash "$0" _balance_worker "${dataset}" >"${LOG_FILE}" 2>&1 &
    printf '%s\n' "$!" >"${PID_FILE}"
    echo "Balance collection started in the background (PID $!)."
    echo "Dataset: ${dataset}"
    echo "Watch it: bash scripts/brev/collect_table_dataset.sh watch"
    ;;
  campaign)
    if running; then
      echo "A collector is already running with PID $(cat "${PID_FILE}")."
      exit 1
    fi
    if pgrep -x kit >/dev/null 2>&1; then
      echo "Isaac Sim is already running. Stop the viewer before collecting."
      echo "Run: bash scripts/brev/stop_fanuc_viewer.sh"
      exit 1
    fi
    mode="${2:-full}"
    case "${mode}" in
      quick) train_target=20; val_target=5; test_target=5 ;;
      full) train_target=100; val_target=25; test_target=25 ;;
      *) echo "Use: $0 campaign [quick|full]"; exit 2 ;;
    esac
    [[ -x "${ISAAC_DIR}/python.sh" ]] || { echo "Isaac Sim not found at ${ISAAC_DIR}"; exit 1; }
    [[ -f "${RUN_DIR}/sort_plan.json" ]] || { echo "Prepared run not found: ${RUN_DIR}"; exit 1; }
    mkdir -p "${JOB_DIR}"
    dataset="${TEAM_DATA_DIR:-/workspace/team-data}/table-campaign-$(date +%Y%m%d-%H%M%S)"
    printf '%s\n' "${dataset}" >"${LATEST_FILE}"
    nohup bash "$0" _campaign_worker "${dataset}" "${train_target}" "${val_target}" "${test_target}" >"${LOG_FILE}" 2>&1 &
    printf '%s\n' "$!" >"${PID_FILE}"
    echo "Balanced campaign started in the background (PID $!)."
    echo "Dataset: ${dataset}"
    echo "Mode ${mode}: $((2 * (train_target + val_target + test_target))) final samples."
    echo "Watch it: bash scripts/brev/collect_table_dataset.sh watch"
    ;;
  status)
    if running; then
      echo "RUNNING (PID $(cat "${PID_FILE}"))"
    else
      echo "NOT RUNNING"
    fi
    [[ -f "${LATEST_FILE}" ]] && echo "Latest dataset: $(cat "${LATEST_FILE}")"
    [[ -f "${LOG_FILE}" ]] && tail -n 12 "${LOG_FILE}"
    ;;
  watch)
    [[ -f "${LOG_FILE}" ]] || { echo "No collection log exists yet."; exit 1; }
    tail -f "${LOG_FILE}"
    ;;
  stop)
    if running; then
      kill -TERM "$(cat "${PID_FILE}")"
      pkill -TERM -x kit 2>/dev/null || true
      echo "Collection stopped. Partial images remain in the latest dataset folder."
    else
      echo "Collector is not running."
    fi
    ;;
  _worker)
    worker "$2" "$3"
    ;;
  _balance_worker)
    balance_worker "$2"
    ;;
  _campaign_worker)
    campaign_worker "$2" "$3" "$4" "$5"
    ;;
  *)
    echo "Usage: $0 {campaign [quick|full]|start [quick|full]|balance|status|watch|stop}"
    ;;
esac
