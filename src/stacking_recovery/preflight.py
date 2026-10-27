"""Fail-closed checks for a frozen grasp-sort-stack demonstration release."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str
    fix: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def is_git_lfs_pointer(path: Path) -> bool:
    """Return true when *path* contains a Git LFS pointer, not model bytes."""
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(len(LFS_POINTER_PREFIX)).startswith(LFS_POINTER_PREFIX)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model(
    path: Path,
    *,
    minimum_bytes: int = 1_000_000,
    expected_sha256: str | None = None,
) -> PreflightCheck:
    if not path.is_file():
        return PreflightCheck(
            "classifier_checkpoint",
            False,
            f"missing: {path}",
            "Run git lfs pull and confirm the configured model path.",
        )
    if is_git_lfs_pointer(path):
        return PreflightCheck(
            "classifier_checkpoint",
            False,
            f"Git LFS pointer found instead of model weights: {path}",
            "Run git lfs pull, then rerun preflight.",
        )
    size = path.stat().st_size
    if size < minimum_bytes:
        return PreflightCheck(
            "classifier_checkpoint",
            False,
            f"checkpoint is unexpectedly small ({size} bytes): {path}",
            "Restore the frozen checkpoint and verify its hash.",
        )
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            return PreflightCheck(
                "classifier_checkpoint",
                False,
                "model_sha256 in the vision config is not a 64-character hex digest",
                "Restore the frozen classifier config.",
            )
        actual = _sha256(path)
        if actual != expected:
            return PreflightCheck(
                "classifier_checkpoint",
                False,
                f"SHA-256 mismatch for {path}: expected {expected}, got {actual}",
                "Run git lfs pull and restore the frozen classifier checkpoint.",
            )
        return PreflightCheck(
            "classifier_checkpoint",
            True,
            f"model present ({size} bytes), SHA-256 verified: {path}",
        )
    return PreflightCheck(
        "classifier_checkpoint", True, f"model present ({size} bytes): {path}"
    )


def check_rl_checkpoint(path: Path) -> PreflightCheck:
    if not path.is_file():
        return PreflightCheck(
            "rl_checkpoint", False, f"missing: {path}", "Restore the frozen Q-table."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return PreflightCheck(
            "rl_checkpoint", False, f"invalid JSON: {error}", "Restore the Q-table."
        )
    if payload.get("format") != "stacking_recovery.q_table.v1":
        return PreflightCheck(
            "rl_checkpoint",
            False,
            f"unsupported format: {payload.get('format')!r}",
            "Use the frozen stacking_recovery.q_table.v1 checkpoint.",
        )
    return PreflightCheck("rl_checkpoint", True, f"valid Q-table: {path}")


def check_file(name: str, path: Path, fix: str) -> PreflightCheck:
    if path.is_file():
        return PreflightCheck(name, True, f"present: {path}")
    return PreflightCheck(name, False, f"missing: {path}", fix)


def check_placement_config(path: Path) -> PreflightCheck:
    """Parse the calibrated placement gate before paying to start Isaac."""
    if not path.is_file():
        return PreflightCheck(
            "placement_config",
            False,
            f"missing: {path}",
            "Restore and calibrate configs/placement_verification.json.",
        )
    try:
        from stacking_recovery.vision.place_check import load_placement_config

        load_placement_config(path)
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return PreflightCheck(
            "placement_config",
            False,
            f"invalid placement config: {error}",
            "Fix the calibrated placement geometry before starting Isaac.",
        )
    return PreflightCheck("placement_config", True, f"valid config: {path}")


def _root_relative(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def check_vision_config(
    path: Path, root: Path
) -> tuple[PreflightCheck, Path | None, str | None]:
    """Validate YAML and resolve its model exactly as the runtime does.

    Runtime model paths are repository-root relative, even when a custom YAML
    file lives elsewhere.  Returning the selected path prevents preflight from
    silently checking an unrelated default checkpoint.
    """
    if not path.is_file():
        return (
            PreflightCheck(
                "vision_config",
                False,
                f"missing: {path}",
                "Restore the frozen classifier configuration.",
            ),
            None,
            None,
        )
    try:
        import yaml
    except ImportError:
        return (
            PreflightCheck(
                "vision_config",
                False,
                "PyYAML is required to validate the classifier config",
                "Install the project ML dependencies before running the demo.",
            ),
            None,
            None,
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        return (
            PreflightCheck(
                "vision_config",
                False,
                f"invalid YAML: {error}",
                "Restore the frozen classifier configuration.",
            ),
            None,
            None,
        )
    if not isinstance(payload, dict):
        return (
            PreflightCheck(
                "vision_config",
                False,
                "classifier config must contain a YAML mapping",
                "Restore the frozen classifier configuration.",
            ),
            None,
            None,
        )
    raw_model = payload.get("model_path")
    if not isinstance(raw_model, str) or not raw_model.strip():
        return (
            PreflightCheck(
                "vision_config",
                False,
                "classifier config is missing model_path",
                "Set model_path to the frozen repository-relative checkpoint.",
            ),
            None,
            None,
        )
    selected = _root_relative(root, Path(raw_model.strip()))
    raw_digest = payload.get("model_sha256")
    digest = str(raw_digest).strip().lower() if raw_digest is not None else None
    hash_note = " with pinned SHA-256" if digest else ""
    return (
        PreflightCheck(
            "vision_config",
            True,
            f"valid config selecting {selected}{hash_note}",
        ),
        selected,
        digest,
    )


def run_preflight(
    root: Path,
    *,
    model_path: Path | None = None,
    rl_path: Path | None = None,
    vision_config: Path | None = None,
    placement_config: Path | None = None,
    run_dir: Path | None = None,
) -> list[PreflightCheck]:
    root = root.resolve()
    config = _root_relative(
        root,
        vision_config or Path("configs/grasp_model_resnet_siamese.yaml"),
    )
    config_check, configured_model, configured_digest = check_vision_config(
        config, root
    )
    alignment_check: PreflightCheck | None = None
    if model_path is not None:
        model = _root_relative(root, model_path)
        expected_digest = configured_digest
        if configured_model is not None and model != configured_model:
            alignment_check = PreflightCheck(
                "classifier_config_alignment",
                False,
                f"requested model {model} does not match config-selected model {configured_model}",
                "Use the model selected by the frozen vision config.",
            )
    else:
        model = configured_model or (root / "models/grasp_pickup_table_resnet18_siamese_224.pt")
        expected_digest = configured_digest
    rl = _root_relative(root, rl_path or Path("models/rl_stacking_qtable.json"))
    checks = [
        config_check,
        check_model(model, expected_sha256=expected_digest),
        check_rl_checkpoint(rl),
    ]
    if alignment_check is not None:
        checks.insert(1, alignment_check)
    if placement_config is not None:
        checks.append(
            check_placement_config(_root_relative(root, placement_config))
        )
    if run_dir is not None:
        source = _root_relative(root, run_dir)
        for name in ("scene_state.json", "sort_plan.json"):
            checks.append(
                check_file(
                    f"run_{name.removesuffix('.json')}",
                    source / name,
                    "Select a complete six-cube run directory.",
                )
            )
    return checks


def passed(checks: Iterable[PreflightCheck]) -> bool:
    return all(check.passed for check in checks)
