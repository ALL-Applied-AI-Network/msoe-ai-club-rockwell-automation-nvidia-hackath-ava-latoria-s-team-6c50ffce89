# Using Our FANUC Digital Twin

This is the quick team guide for our shared FANUC ER-4iA simulation on Brev. It
is meant to help you get in, run your part of the project, and avoid burning GPU
credits by accident.

## What the digital twin is for

Our simulated workcell has the FANUC arm, gripper, table, cameras, cubes, caps,
and tubes. The final task is **cube-only stacking**: the cubes are targets, while
the caps and tubes are distractors the system should ignore. The planned stack
order is yellow, yellow, blue, blue, red, red.

The digital twin lets us test motion, collect images, train/check vision models,
and evaluate recovery logic before anyone tries it on the real robot.

## Before you start

- Use the shared environment instead of making another paid GPU instance.
- Only one person should use the Isaac browser viewer at a time.
- Ask in the team chat before rebuilding the scene or collecting a large batch.
- Work on a branch and do not commit generated images, videos, model weights, or
  the generated `fanuc_cell.usd` file.

## Get into the shared Brev environment

1. Open the [team invite link](https://brev.nvidia.com/invite?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbnZpcm9ubWVudElkIjoiIiwiZXhwaXJhdGlvbiI6MTc5MzA0MTk4NCwib3JnSWQiOiJvcmctM0thMGx4dTZtWVJubVBNc0V4bzRORUV5TmxLIiwib3JnYW5pemF0aW9uQWNjZXNzTGV2ZWwiOiJPcmdhbml6YXRpb25NZW1iZXIiLCJ1c2VySWQiOiJ1c2VyLTNLdlRaNW12MGVkQjhqbUpzOTRoWE1kVk50QyJ9.XVhA976QowyKncx64Qni_7WeadPOEoE0NNuyQfeAk3w&orgID=org-3Ka0lxu6mYRnmPMsExo4NEEyNlK)
   and sign in or create your Brev account.
2. In Brev, open **Environments**.
3. Click **fanuc-digital-twin-v5**. This is the shared environment; do not make
   another GPU environment.
4. If it says **Stopped**, click **Start** and wait until it is running.
5. Click the **Access** tab.
6. Open the Jupyter link. A JupyterLab page will open in a new tab.
7. In JupyterLab's top menu, click **File**, then **New**, then **Terminal**. A
   terminal tab with a command prompt will open. Paste commands there one line
   at a time and press **Enter** after each one.

## Find the repo inside Brev

The actual GitHub repository is named
`msoe-ai-club-rockwell-automation-nvidia-hackath-ava-latoria-s-team-6c50ffce89`.
In our current shared Brev environment it is stored in the shorter local folder
`/workspace/team-repo`. Use this command to open it:

```bash
cd /workspace/team-repo
git status
```

If both commands work, the repo is already there. If the first command says the
folder does not exist, install the repo in the same Brev location with:

```bash
cd /workspace
git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-ava-latoria-s-team-6c50ffce89.git team-repo
cd /workspace/team-repo
git switch main
```

GitHub may ask you to sign in because the repository is private. The source code
will be in `/workspace/team-repo`. The generated scene, logs, and shared data are
kept separately in `/workspace/team-data` so they do not get committed to Git.

## Open the twin and watch the arm

1. Follow the steps above to open a Jupyter terminal.
2. Paste and run these commands one line at a time:

   ```bash
   cd /workspace/team-repo
   git switch main
   git pull
   bash scripts/brev/start_fanuc_viewer.sh
   ```

3. Wait about a minute. To watch the startup messages, run:

   ```bash
   tail -f /workspace/team-data/fanuc-stream.log
   ```

   When it says `safe looping demonstration started`, press **Ctrl+C**. This only
   stops watching the messages; it does not stop Isaac Sim.
4. Return to the Brev environment's **Access** page and open the Isaac viewer.
   The current viewer is
   [isaac-ya9khmx83.gobrev.dev/viewer](https://isaac-ya9khmx83.gobrev.dev/viewer/),
   but use the Access page if Brev ever changes the link. Keep only one viewer
   tab open. The arm should slowly move through safe poses and operate its
   gripper, just like the working demonstration.

To stop only the viewer:

```bash
bash scripts/brev/stop_fanuc_viewer.sh
```

If the shared stage is ever missing, rebuild it once:

```bash
bash scripts/brev/build_fanuc_stage.sh
```

The first build can take several minutes because Isaac may download the FANUC
asset. Do not run the build while the viewer is running.

## Where shared data goes

Run this once in the same Jupyter terminal to create the shared data settings:

```bash
cd /workspace/team-repo
bash scripts/brev/bootstrap_data_workspace.sh
source /workspace/team-data.env
```

The important locations are:

- `/workspace/team-data/fanuc_cell.usd` - generated workcell used by Isaac.
- `/workspace/team-data/fanuc-stream.log` - viewer and motion log.
- `/workspace/team-data/grasp_quality` - shared wrist images and labels.
- `results/runs/<run-name>` - one run's plan, frames, events, and report.

The workspace disk survives a normal Brev stop. Deleting the environment can
delete that disk, so copy important results before deleting anything.

## What each team role can do with it

### Data and simulation images

Use the table and wrist cameras to collect examples. Change one thing at a time,
such as lighting, cube position, partial blockage, or a slightly bad grasp. Keep
train, validation, and test images separate and update the manifest.

### Model training and evaluation

Use the shared images, but you do not need the Isaac viewer running while a model
trains. Start with the repo's PyTorch/MobileNetV3 path, save the class order and
preprocessing settings, and evaluate on the frozen test split.

### Runtime vision integration

Turn one camera frame into the shared result: outcome label, confidence,
recommended action, and a short reason. Test the interface on saved images before
connecting it to a live simulation run.

### Recovery policy

Most policy work can run locally without a GPU. Use saved or mocked vision
results to test continue, retry, regrasp, reacquire, and safe-stop decisions.

### Simulation and robot sequence

Use the twin to connect plan, grasp, verify, recover, place, and verify again.
Start from the safe motion demo, then test one cube before attempting a full
stack. Keep the real FANUC completely separate and supervised.

### Testing and demo evidence

Reuse the same starting cases for open loop, rules, and AI. Save the run ID,
result report, and a short clip or screenshot. Failed runs are useful evidence if
the reason is recorded honestly.

## Common problems

- **Waiting for stream:** wait for the ready message, close duplicate viewer
  tabs, and reopen the link from Brev's Access page.
- **Empty grid:** the stage did not load. Check `fanuc-stream.log`, confirm
  `fanuc_cell.usd` exists, and restart the viewer.
- **Arm is visible but still:** confirm the log contains `[fanuc-demo] moving`.
- **Viewer says busy:** another viewer tab or teammate is connected.
- **A script says AWS or asset error:** check whether the final scene still
  exported. A warning is not automatically a failed run.
- **Credits dropping:** stop the whole Brev environment when nobody is using the
  GPU. Closing the browser does not stop billing.

## Safety rule

This simulation is not permission to run the real FANUC. Real-robot motion,
controller changes, and gripper changes must be supervised by the event's FANUC
or Rockwell contact. Test in Isaac first and use a safe stop when anything is
uncertain.
