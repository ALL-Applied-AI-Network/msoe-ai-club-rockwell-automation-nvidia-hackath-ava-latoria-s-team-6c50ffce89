# Allowed Outcomes and actions

|Outcome| Situation                                                             | Action     |
|---|-----------------------------------------------------------------------|------------|
|SUCCESS| Seated/placed with confidence, greater or equal to the confidence min | continue   |
|LOW_CONFIDENCE| Low confidence with retries left                                      | retry      |
|OBJECT_NOT_FOUND| Cube or table not visible or low table pose confidence                | reaquire   |
|GRASP_FAILED| Cube not seated properly with reties left                             | regrasp    |
|INVALID| Invaldi input| safe_stop  |
|All retries used||  safe_stop |