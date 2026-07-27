# Orientation tests

Controlled trajectories for verifying model pitch/yaw/roll behavior.

- `gen_orientation_tests.py` — writes straight-line paths that sweep one Euler
  axis at a time into `orientation/` (isolates each column).
- `render_orientation_tests.py <out_dir>` — renders start/mid/end frames of each
  test so the model's rotation can be read off directly.

Angles are in the aerospace convention (yaw compass/clockwise-positive, pitch
nose-up-positive) — the default. Add `--math` on the CLI for the math
convention (yaw CCW-positive), or `--flip-yaw/--flip-pitch/--flip-roll` to
correct a single axis for data in another convention.
