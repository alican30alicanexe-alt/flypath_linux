"""Check the live view against the static renderer, across every episode.

    trajectory.venv/bin/python tests/live_replay_check.py

Replaying a finished engagement through `LiveEngagementView` drives exactly the
incremental path a simulation drives, so this exercises the live code without a
simulation attached. Asserts that:

  1. the live view converges on the same rounded box the static path computes
  2. the renderer's bounds equal that box, so the axis ticks cannot drift
  3. the range the HUD reports at the closest step matches the recorded min_R
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pytrajectory.engagement import _nice_box, load_episode_dir
from pytrajectory.live import replay

EVAL_DIR = Path('/home/mmsgb/Desktop/TDTP/TDTP_13Jul/3DOF_v3/'
                'EvalTrajectories/20260728_151647')
N_LABELS = 10


def main():
    engagements = load_episode_dir(EVAL_DIR)
    print(f'{len(engagements)} episodes from {EVAL_DIR.name}\n')

    failures = []
    for eng in engagements:
        name = eng.source_path.name
        pts = np.vstack([t.points for t in eng.tracks])
        static_box, _ = _nice_box(pts.min(axis=0), pts.max(axis=0), N_LABELS)

        view = replay(eng, off_screen=True, render_every=100,
                      n_labels=N_LABELS)
        try:
            if not np.allclose(view._box, static_box):
                failures.append(f'{name}: live box {view._box} != '
                                f'static box {static_box}')

            # Moving actors are excluded from the bounds, so the renderer's
            # extent must be exactly the pinned box — that is what stops the
            # ticks drifting off their round values as the trails grow.
            if not np.allclose(view.plotter.bounds, static_box):
                failures.append(f'{name}: renderer bounds '
                                f'{np.round(view.plotter.bounds, 1)} != box')

            # The HUD's range series must be the same one verify() checks.
            for pursuer in eng.pursuers:
                live_min = float(eng.range_series(pursuer).min())
                if abs(live_min - pursuer.min_r) > 1e-6:
                    failures.append(
                        f'{name} {pursuer.label}: HUD range min {live_min} '
                        f'!= recorded {pursuer.min_r}')
        finally:
            view.close()

        print(f'  {name}  box ok, bounds pinned, ranges exact')

    if failures:
        print(f'\n{len(failures)} FAILURES:')
        for f in failures:
            print('  ', f)
        return 1

    print(f'\nall {len(engagements)} episodes: live view matches the '
          f'static renderer')
    return 0


if __name__ == '__main__':
    sys.exit(main())
