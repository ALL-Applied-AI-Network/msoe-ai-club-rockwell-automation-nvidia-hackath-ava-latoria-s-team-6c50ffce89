#!/usr/bin/env python3
"""Write a no-motion hardware command transcript from a sort/stack plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.hardware import (  # noqa: E402
    ShadowRobotAdapter,
    execute_shadow_plan,
    load_hardware_adapter,
)


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="sort_plan.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "hardware" / "fanuc_schunk_template.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clearance-m", type=float, default=0.05)
    parser.add_argument("--speed-scale", type=float, default=0.10)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    adapter = load_hardware_adapter(args.config.resolve())
    if not isinstance(adapter, ShadowRobotAdapter):
        raise SystemExit("hardware shadow runner requires config mode='shadow'")
    transcript = execute_shadow_plan(
        plan,
        adapter,
        clearance_m=args.clearance_m,
        speed_scale=args.speed_scale,
    )
    transcript["source_plan"] = str(args.plan.resolve())
    transcript["hardware_config"] = str(args.config.resolve())
    write_json_atomic(args.out.resolve(), transcript)
    print(f"Wrote {len(transcript['commands'])} no-motion commands to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
