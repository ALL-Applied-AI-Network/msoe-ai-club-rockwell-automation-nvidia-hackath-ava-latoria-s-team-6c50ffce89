#!/usr/bin/env python3
"""Validate frozen demo inputs before starting Isaac Sim or robot adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.preflight import passed, run_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--rl-checkpoint", type=Path)
    parser.add_argument("--vision-config", type=Path)
    parser.add_argument(
        "--placement-config",
        type=Path,
        default=ROOT / "configs" / "placement_verification.json",
    )
    parser.add_argument("--run", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args()
    checks = run_preflight(
        args.root,
        model_path=args.model,
        rl_path=args.rl_checkpoint,
        vision_config=args.vision_config,
        placement_config=args.placement_config,
        run_dir=args.run,
    )
    if args.json:
        print(json.dumps({"passed": passed(checks), "checks": [c.to_dict() for c in checks]}, indent=2))
    else:
        for check in checks:
            print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
            if check.fix and not check.passed:
                print(f"       Fix: {check.fix}")
    return 0 if passed(checks) else 2

if __name__ == "__main__":
    raise SystemExit(main())
