import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grasp_sort_scorecard.py"
SPEC = importlib.util.spec_from_file_location("grasp_sort_scorecard", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pick(*, retry: int = 0, place_retry: int = 0):
    return {
        "pickup_verified": True,
        "placement_verified": True,
        "placement_error_mm": 5.0,
        "cycle_time_s": 12.0,
        "closed_loop_grasp_attempts": retry,
        "closed_loop_placement_attempts": place_retry,
    }


def test_ready_after_twenty_good_picks_including_recovery():
    picks = [_pick() for _ in range(19)] + [_pick(retry=1)]
    result = MODULE.build_scorecard([{"picks": picks}])
    assert result["decision"] == "GO_TO_STACKING"
    assert result["successful_recoveries"] == 1


def test_missing_recovery_evidence_keeps_sort_gate_closed():
    result = MODULE.build_scorecard([{"picks": [_pick() for _ in range(20)]}])
    assert result["decision"] == "KEEP_TESTING_SORT"
    assert result["checks"]["recovery"] is False


def test_pick_rate_counts_retries_as_attempts():
    result = MODULE.build_scorecard([{"picks": [_pick(retry=1)]}])
    assert result["pick_attempts"] == 2
    assert result["pick_success_per_attempt"] == 0.5
