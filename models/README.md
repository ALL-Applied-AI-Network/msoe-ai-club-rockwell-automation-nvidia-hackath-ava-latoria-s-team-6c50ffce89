# Grasp-quality checkpoints

Train locally (no Isaac):

```bash
pip install -e ".[ml]"
python scripts/generate_procedural_wrist_dataset.py
python scripts/train_grasp_classifier.py
```

Weights land at `models/grasp_quality_mobilenet_v3.pt` (gitignored). Copy to Brev or commit via LFS if the team shares one checkpoint.

Sim closed loop with AI:

```bash
CLOSED_LOOP=1 VERIFICATION=ai TRIAL_CONFIG=... ./run_sort_pipeline.sh
```

For demonstrations before the classifier has been validated on a representative
Isaac or real-camera holdout set, use `VERIFICATION=ai_guarded`. This preserves
the learned failure class for recovery but allows motion only when the AI and
the deterministic wrist check both accept the grasp.
