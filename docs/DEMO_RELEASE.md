# Freezing the demo release

Freeze a release only after the classifier holdout and matched Isaac campaign
produce `results/lab_readiness/readiness.json` with `"overall": "go"`. The
freezer checks the model, RL checkpoint, configs, source run and Git state,
then records SHA-256 hashes for every frozen input:

```bash
python scripts/freeze_demo_release.py \
  --run /workspace/team-data/grasp-sort-live/sort_baseline_full_02
```

The qualified manifest is written to
`results/release/release_manifest.json`. Review it, then commit/tag/push the
already-tested source using the team's normal Git process. The freezer does
not change Git, start Isaac Sim, or connect to hardware.

By default, a dirty worktree, failed preflight, missing evidence or a readiness
result other than `GO` stops the command without writing a manifest. For local
development only, `--allow-dirty` and `--skip-readiness` are available. Either
override produces a visibly `development_only` manifest that must not be used
as lab qualification evidence.

Before the lab, copy the qualified manifest, readiness report, videos and run
logs to an offline folder. Re-run preflight after every clone or `git lfs pull`
and verify that its hashes match the frozen manifest.
