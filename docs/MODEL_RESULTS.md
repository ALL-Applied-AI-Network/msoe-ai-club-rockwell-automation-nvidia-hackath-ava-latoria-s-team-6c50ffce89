# Grasp model results

The current recommended pickup verifier is a ResNet-18 Siamese model. It compares the full RGB image before pickup with the full RGB image after pickup instead of compressing the pair into a grayscale difference image.

## Best result

- Dataset: 2,520 Isaac Sim before/after image pairs
- Fixed held-out test set: 180 pairs
- Test accuracy: **99.44% (179/180)**
- Macro-F1: **99.44%**
- Unsafe GOOD rate: **1.11%**
- Test confusion matrix: `[[90, 0], [1, 89]]`
- Inference time on the Brev L4: about **0.55 ms per pair**
- Decision threshold: `tau_good: 0.525`, chosen on validation data only

The single 224 px ResNet-18 model beat the tested ensemble, so the simpler single-model runtime is used.

## Files

- Runtime config: `configs/grasp_model_resnet_siamese.yaml`
- Checkpoint: `models/grasp_pickup_table_resnet18_siamese_224.pt`
- Training/benchmark runner: `scripts/benchmark_grasp_models.py`
- Graph generator: `scripts/plot_grasp_benchmark.py`
- Metrics and graphs: `results/grasp_benchmark/`

The checkpoint is stored with Git LFS. After cloning, run `git lfs pull` before using the model.

## Recreate the benchmark

Run the benchmark in the Brev/Isaac environment. Set the dataset path to the folder containing `manifest.csv`, then use zero data-loader workers on this VM to avoid its small shared-memory limit.

```bash
PYTHONPATH=src /workspace/isaaclab/_isaac_sim/python.sh scripts/benchmark_grasp_models.py \
  --dataset /workspace/team-data/table-campaign-20261022-203838-augmented \
  --output /workspace/model-search \
  --workers 0
```

Create the evaluation figures from the saved JSON files:

```bash
PYTHONPATH=src /workspace/isaaclab/_isaac_sim/python.sh scripts/plot_grasp_benchmark.py \
  --results /workspace/model-search \
  --winner /workspace/model-search/resnet18_siamese.json \
  --output results/grasp_benchmark
```

## Important limitation

These numbers measure a held-out split from the same Isaac Sim collection campaign. They show that the model learned this simulated task well, but they do not prove the same accuracy on the physical FANUC cell. Treat the model as a verifier, keep the deterministic state-machine safety checks, and validate under supervised robot operation before relying on it.
