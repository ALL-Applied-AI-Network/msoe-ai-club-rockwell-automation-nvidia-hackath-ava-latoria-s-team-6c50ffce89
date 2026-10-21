#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate the binary table-camera pickup verifier on the most recent
# shared dataset. Run inside the Brev Jupyter terminal (vscode container).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TEAM_REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ISAAC_DIR="${ISAAC:-/isaac-sim}"
LATEST_FILE="${TEAM_DATA_DIR:-/workspace/team-data}/table-collection-job/latest-dataset.txt"

[[ -f "${LATEST_FILE}" ]] || { echo "No collected dataset is recorded."; exit 1; }
DATASET="$(cat "${LATEST_FILE}")"
[[ -f "${DATASET}/manifest.json" ]] || { echo "Dataset not found: ${DATASET}"; exit 1; }

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src"

echo "Validating ${DATASET}"
"${ISAAC_DIR}/python.sh" scripts/validate_grasp_dataset.py --data "${DATASET}"

echo "Training the table-camera pickup verifier"
"${ISAAC_DIR}/python.sh" scripts/train_grasp_classifier.py \
  --data "${DATASET}" \
  --config configs/grasp_model_table.yaml \
  --observation-source table_before_after \
  --fast

mkdir -p "${DATASET}/evaluation"
for split in val test; do
  echo "Evaluating ${split} split"
  "${ISAAC_DIR}/python.sh" scripts/eval_grasp_classifier.py \
    --data "${DATASET}" \
    --config configs/grasp_model_table.yaml \
    --split "${split}" \
    --observation-source table_before_after \
    --json-out "${DATASET}/evaluation/table-pickup-${split}.json"
done

echo "TRAINING COMPLETE"
echo "Model: ${REPO_DIR}/models/grasp_pickup_table_mobilenet_v3.pt"
echo "Results: ${DATASET}/evaluation"
