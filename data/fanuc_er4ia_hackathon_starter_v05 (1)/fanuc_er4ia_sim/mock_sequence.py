import numpy as np

from stacking_recovery.types import (
    FailureReason,
    GraspCheckResult,
    PlaceCheckResult,
)
from stacking_recovery.vision.grasp_check import grasp_check_stub
from stacking_recovery.vision.place_check import place_check_stub


def run_mock_sequence(grasp_success, placement_success):
    fake_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    stack_level = 0  # First cube in the stack.

    phases = [
        "approach", "descend", "close", "dwell",
        "lift", "carry", "lower", "release",
        "settle", "retreat", "home",
    ]

    for phase in phases:
        print(f"Phase: {phase}")

        if phase == "dwell":
            grasp_result = grasp_check_stub(
                fake_image,
                scripted=GraspCheckResult(
                    seated=grasp_success,
                    yaw_error_deg=0.0,
                    confidence=1.0,
                    reason=(
                        FailureReason.OK
                        if grasp_success
                        else FailureReason.EMPTY_GRASP
                    ),
                ),
            )

            print(f"Grasp successful: {grasp_result.seated}")

            if not grasp_result.seated:
                print("Stopping before lift.")
                return

        if phase == "settle":
            placement_result = place_check_stub(
                fake_image,
                stack_level,
                scripted=PlaceCheckResult(
                    placed=placement_success,
                    confidence=1.0,
                    reason=(
                        FailureReason.OK
                        if placement_success
                        else FailureReason.PLACEMENT_FAILURE
                    ),
                ),
            )

            print(f"Placement successful: {placement_result.placed}")

            if not placement_result.placed:
                print("Placement failed. Stopping mock sequence.")
                return

    print("Mock sequence completed.")


if __name__ == "__main__":
    print("--- Successful grasp and placement ---")
    run_mock_sequence(grasp_success=True, placement_success=True)

    print("\n--- Failed grasp ---")
    run_mock_sequence(grasp_success=False, placement_success=True)

    print("\n--- Failed placement ---")
    run_mock_sequence(grasp_success=True, placement_success=False)