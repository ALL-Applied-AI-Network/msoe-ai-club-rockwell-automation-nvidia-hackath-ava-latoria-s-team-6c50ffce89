# Brev and Isaac Sim helpers

Start with [`../../docs/DIGITAL_TWIN_GUIDE.md`](../../docs/DIGITAL_TWIN_GUIDE.md).
It covers the shared environment, browser viewer, motion demo, team workflows,
troubleshooting, GPU credits, and safety.

The scripts in this folder provide repeatable commands for the shared setup:

- `build_fanuc_stage.sh` creates the generated FANUC workcell USD.
- `start_fanuc_viewer.sh` starts streaming with the safe motion demo.
- `stop_fanuc_viewer.sh` stops the streaming process.
- `bootstrap_data_workspace.sh` creates the shared data paths below.

## Shared data workspace

Use one persistent folder on the Brev instance for Isaac images and labels. Do
not run multiple collectors at once because they append to one manifest.

```bash
cd /home/ubuntu/workspace/team-repo
bash scripts/brev/bootstrap_data_workspace.sh
source /home/ubuntu/workspace/team-data.env
```

This only prepares the shared folder. Each teammate's feature branch supplies
its own collection, training, recovery, or evaluation command; those delegated
implementations are intentionally not part of the shared `main` setup.

The instance and dataset become available to teammates after they join the same
Brev organization. Stop the instance when nobody is collecting or training;
files in `/home/ubuntu/workspace` remain on its persistent disk.
