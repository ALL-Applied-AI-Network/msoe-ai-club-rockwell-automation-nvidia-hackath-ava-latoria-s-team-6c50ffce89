# Brev data workspace

Use one persistent folder on the Brev instance for Isaac images and labels. Do
not run multiple collectors at once because they append to one manifest.

```bash
cd /home/ubuntu/workspace/team-repo
bash scripts/brev/bootstrap_data_workspace.sh
source /home/ubuntu/workspace/team-data.env
python scripts/validate_grasp_dataset.py
```

Collect every prepared run sequentially:

```bash
$ISAAC/python.sh scripts/sim/collect_wrist_dataset_batch.py \
  --repo "$TEAM_REPO_DIR" \
  --runs-dir "$TEAM_REPO_DIR/results/runs"
```

Train or evaluate from the same shared folder:

```bash
python scripts/train_grasp_classifier.py
python scripts/eval_grasp_classifier.py
```

The instance and dataset become available to teammates after they join the same
Brev organization. Stop the instance when nobody is collecting or training;
files in `/home/ubuntu/workspace` remain on its persistent disk.
