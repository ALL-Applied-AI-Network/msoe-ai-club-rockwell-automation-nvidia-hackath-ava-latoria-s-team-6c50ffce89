# Brev and Isaac Sim helpers

Start with [`../../docs/DIGITAL_TWIN_GUIDE.md`](../../docs/DIGITAL_TWIN_GUIDE.md).
It covers the shared environment, browser viewer, motion demo, team workflows,
troubleshooting, GPU credits, and safety.

The scripts in this folder provide repeatable commands for the shared setup:

- `build_fanuc_stage.sh` creates the generated FANUC workcell USD.
- `start_fanuc_viewer.sh` starts streaming with the safe motion demo.
- `stop_fanuc_viewer.sh` stops the streaming process.
- `bootstrap_data_workspace.sh` creates the shared data paths below.
- `collect_table_dataset.sh` runs labeled table-camera collection in the background.

## Easy background collection

For the model-quality collection, use the balanced campaign. It checks the
labels already saved after every simulated attempt and keeps generating varied
nominal, shifted, rotated, and empty-grasp conditions until both binary classes
reach their train, validation, and test quotas:

```bash
cd /workspace/team-repo-ava-049bfe8
bash scripts/brev/stop_fanuc_viewer.sh
bash scripts/brev/collect_table_dataset.sh campaign full
```

`full` targets 300 final samples: 100 GOOD + 100 failures in train, then 25 +
25 in validation and test. Use `campaign quick` for a 60-sample smoke test.
Every attempt uses a fresh seed and cycles cube targets and failure types. The
actual physics result decides the saved label, so a failure requested by the
planner is never falsely counted as a failure if the robot succeeds.

Use `status` or `watch` below to monitor it. The job survives closing your
laptop, but the Brev environment must remain running until it completes.

### Older fixed collection

Open a terminal in the Brev Jupyter page, then run:

```bash
cd /workspace/team-repo-ava-049bfe8
bash scripts/brev/stop_fanuc_viewer.sh
bash scripts/brev/collect_table_dataset.sh start quick
```

`quick` collects two attempts from each of five train/validation/test conditions
(10 samples total). Use `full` instead to collect all six cube attempts per
condition (30 samples total). Closing your laptop does not stop
the background job, but the Brev GPU still uses credits while the environment is
running.

Check progress without starting another collector:

```bash
cd /workspace/team-repo
bash scripts/brev/collect_table_dataset.sh status
```

To watch the live log, run the command below. Press `Ctrl+C` to stop watching;
the collection itself keeps running.

```bash
bash scripts/brev/collect_table_dataset.sh watch
```

If an initial collection completes but the train/test labels are one-sided,
fill the missing coverage without replacing the existing dataset:

```bash
bash scripts/brev/collect_table_dataset.sh balance
bash scripts/brev/collect_table_dataset.sh watch
```

After the log says `BALANCE COLLECTION COMPLETE`, train and evaluate the
binary pickup verifier:

```bash
bash scripts/brev/train_table_pickup_model.sh
```

The last log line says `COLLECTION COMPLETE` when it is done. Then stop the
environment from the Brev dashboard so billing stops. If you need to cancel the
collection early, use:

```bash
bash scripts/brev/collect_table_dataset.sh stop
```

## Shared data workspace

Use one persistent folder on the Brev instance for Isaac images and labels. Do
not run multiple collectors at once because they append to one manifest.

```bash
cd /workspace/team-repo-ava-049bfe8
bash scripts/brev/bootstrap_data_workspace.sh
source /home/ubuntu/workspace/team-data.env
```

This only prepares the shared folder. Each teammate's feature branch supplies
its own collection, training, recovery, or evaluation command; those delegated
implementations are intentionally not part of the shared `main` setup.

The instance and dataset become available to teammates after they join the same
Brev organization. Stop the instance when nobody is collecting or training;
files in `/home/ubuntu/workspace` remain on its persistent disk.
