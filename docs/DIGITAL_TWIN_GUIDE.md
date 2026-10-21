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

- Join the team Brev organization as a member.
- Use the shared environment instead of making another paid GPU instance.
- Only one person should use the Isaac browser viewer at a time.
- Ask in the team chat before rebuilding the scene or collecting a large batch.
- Work on a branch and do not commit generated images, videos, model weights, or
  the generated `fanuc_cell.usd` file.

## Open the twin and watch the arm

1. Start the shared Brev environment if it is stopped.
2. Open its Jupyter link and launch a terminal.
3. Run:

   ```bash
   cd /workspace/team-repo
   git pull
   bash scripts/brev/start_fanuc_viewer.sh
   ```

4. Wait about a minute, then check:

   ```bash
   tail -f /workspace/team-data/fanuc-stream.log
   ```

5. When the log says `safe looping demonstration started`, open the Isaac
   viewer link from the Brev environment's Access page. Keep only one viewer tab
   open. The arm should slowly move through safe poses and operate its gripper.

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

Run this once in the Brev host terminal to create the shared data settings:

```bash
cd /home/ubuntu/workspace/team-repo
bash scripts/brev/bootstrap_data_workspace.sh
source /home/ubuntu/workspace/team-data.env
```

The important locations are:

- `/workspace/team-data/fanuc_cell.usd` - generated workcell used by Isaac.
- `/workspace/team-data/fanuc-stream.log` - viewer and motion log.
- `/home/ubuntu/workspace/team-data/grasp_quality` - shared wrist images and labels.
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
