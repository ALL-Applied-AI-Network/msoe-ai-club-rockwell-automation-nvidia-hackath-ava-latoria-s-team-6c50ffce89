# Stretch: alternative perception front end

**Not part of the primary deliverable.** Start only after closed-loop cube stacking and the benchmark table are frozen.

Goal: lightweight detector for cubes, caps, and tubes. The recovery policy in `src/stacking_recovery/control/` must **not** import YOLO — only the table-camera planner swaps detectors.

Claim if demonstrated: *"The recovery framework is object-agnostic and works with alternative perception front ends."*
