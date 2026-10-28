#!/usr/bin/env python3
"""Create a hash-locked manifest for a qualified demo release.

This tool records evidence only.  It never creates a Git tag, pushes a branch,
starts Isaac Sim, or connects to robot hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.preflight import passed, run_preflight  # noqa: E402


class ReleaseGateError(RuntimeError):
    """Raised when evidence is incomplete or unsafe to freeze."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise ReleaseGateError(detail)
    return result.stdout.strip()


def git_snapshot(root: Path) -> dict[str, Any]:
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root.resolve():
        raise ReleaseGateError(f"--root must be the Git repository root: {top}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    changed = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current") or "detached",
        "clean": not changed,
        "changed_paths": changed,
    }


def file_evidence(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ReleaseGateError(f"release input is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        display_path = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_readiness(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReleaseGateError(f"lab-readiness evidence is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseGateError(f"lab-readiness evidence is invalid: {error}") from error
    if str(payload.get("overall", "")).strip().lower() != "go":
        raise ReleaseGateError(
            "lab-readiness evidence is not GO; run the acceptance campaign first"
        )
    return payload


def build_manifest(
    *,
    root: Path,
    run_dir: Path,
    model: Path,
    rl_checkpoint: Path,
    vision_config: Path,
    placement_config: Path,
    readiness_path: Path | None,
    allow_dirty: bool = False,
    skip_readiness: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    run_dir = run_dir.resolve()
    checks = run_preflight(
        root,
        model_path=model.resolve(),
        rl_path=rl_checkpoint.resolve(),
        vision_config=vision_config.resolve(),
        placement_config=placement_config.resolve(),
        run_dir=run_dir,
    )
    if not passed(checks):
        failed = "; ".join(
            f"{check.name}: {check.detail}" for check in checks if not check.passed
        )
        raise ReleaseGateError(f"preflight failed: {failed}")

    git = git_snapshot(root)
    if not git["clean"] and not allow_dirty:
        raise ReleaseGateError(
            "Git worktree is dirty; commit/stash the listed changes or use "
            "--allow-dirty for a non-qualified development manifest"
        )

    readiness: dict[str, Any]
    readiness_evidence: dict[str, Any] | None = None
    if skip_readiness:
        readiness = {
            "status": "skipped",
            "warning": "acceptance evidence was explicitly skipped",
        }
    else:
        if readiness_path is None:
            raise ReleaseGateError("--readiness is required unless --skip-readiness is set")
        payload = load_readiness(readiness_path.resolve())
        readiness_evidence = file_evidence(readiness_path, root)
        readiness = {
            "status": "go",
            "overall": payload["overall"],
            "evidence": readiness_evidence,
        }

    # Supplying either override is itself evidence that this was not the strict
    # qualification path, even when the worktree happens to be clean.
    development_only = allow_dirty or skip_readiness
    files = {
        "classifier": file_evidence(model, root),
        "rl_checkpoint": file_evidence(rl_checkpoint, root),
        "vision_config": file_evidence(vision_config, root),
        "placement_config": file_evidence(placement_config, root),
        "scene_state": file_evidence(run_dir / "scene_state.json", root),
        "sort_plan": file_evidence(run_dir / "sort_plan.json", root),
    }
    ground_truth = run_dir / "ground_truth.json"
    if ground_truth.is_file():
        files["ground_truth"] = file_evidence(ground_truth, root)

    return {
        "schema_version": "stacking_recovery.demo_release.v1",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "qualification": "development_only" if development_only else "qualified",
        "qualified": not development_only,
        "git": git,
        "preflight": [check.to_dict() for check in checks],
        "readiness": readiness,
        "files": files,
        "safety": {
            "creates_git_tag": False,
            "pushes_repository": False,
            "starts_simulator": False,
            "connects_to_hardware": False,
        },
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "grasp_pickup_table_resnet18_siamese_224.pt",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        default=ROOT / "models" / "rl_stacking_qtable.json",
    )
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=ROOT / "configs" / "grasp_model_resnet_siamese.yaml",
    )
    parser.add_argument(
        "--placement-config",
        type=Path,
        default=ROOT / "configs" / "placement_verification.json",
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=ROOT / "results" / "lab_readiness" / "readiness.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "release" / "release_manifest.json",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="write a development-only manifest from an intentionally dirty worktree",
    )
    parser.add_argument(
        "--skip-readiness",
        action="store_true",
        help="write a development-only manifest without lab-readiness evidence",
    )
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(
            root=args.root,
            run_dir=args.run,
            model=args.model,
            rl_checkpoint=args.rl_checkpoint,
            vision_config=args.vision_config,
            placement_config=args.placement_config,
            readiness_path=args.readiness,
            allow_dirty=args.allow_dirty,
            skip_readiness=args.skip_readiness,
        )
    except ReleaseGateError as error:
        print(f"RELEASE NOT FROZEN: {error}", file=sys.stderr)
        return 2
    write_json_atomic(args.output.resolve(), manifest)
    print(
        f"Wrote {manifest['qualification']} release manifest to "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
