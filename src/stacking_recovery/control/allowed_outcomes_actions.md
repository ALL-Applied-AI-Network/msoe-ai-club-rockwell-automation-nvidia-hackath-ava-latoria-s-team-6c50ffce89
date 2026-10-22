# Allowed Outcomes and actions


| Outcome         | Situation                                                     | Action    |
|-----------------|---------------------------------------------------------------|-----------|
| OK              | Cube was successfully placed with >= 0.6 confidence           | PROCEED   |
| EMPTY_GRASP     | Gripper failed to pick up the cube                            | REGRASP   |
| MISALIGNED_CUBE | Cube is not properly aligned in the gripper                   | REGRASP   |
| OCCLUSION       | Camera view is blocked, so there is not reliable verification | REACQUIRE |
| LOW_CONFIDENCE  | Vision confidence is below 0.6                                | RETRY     |
| TITLED_OBJECT   | Cube is tilted or incorreclty oriented/placed                 | REGRASP   |
|PLACEMENT_FAILURE| Cube was not placed correlty                                  | RETRY     |
|UNKNOWN| Vision system cannot identify the failure                     | SAFE_STOP |
|STUB| Placeholder result rather than verified vision output         | SAFE_STOP          |
|Invalid input| Invalid confidence, missing data, or malformed vision result  | SAFE_STOP          |
|Retry limit reached| Maximum allowed recovery attepts reached                      | SAFE_STOP          |

MAX_GRASP_RETRIES = 2
MAX_PLACEMENT_RETRIES = 1