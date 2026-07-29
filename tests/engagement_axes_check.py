"""Checks on the engagement axis fitting, across every exported episode.

    trajectory.venv/bin/python tests/engagement_axes_check.py

Asserts three things that must hold for every episode:
  1. the drawn box contains all data — tightening never crops a sample
  2. every tick lands on an exact multiple of its axis's round step
  3. the geometry still agrees with the simulation's recorded min_R
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pytrajectory.engagement import _nice_box, load_episode_dir

EVAL_DIR = Path('/home/mmsgb/Desktop/TDTP/TDTP_13Jul/3DOF_v3/'
                'EvalTrajectories/20260728_151647')
N_LABELS = 10


def main():
    if not EVAL_DIR.is_dir():
        print(f'Export directory missing: {EVAL_DIR}\n'
              f'Regenerate it with eval_v4.py, or point EVAL_DIR at another '
              f'export. Skipping.')
        return 0

    engagements = load_episode_dir(EVAL_DIR)
    print(f'{len(engagements)} episodes from {EVAL_DIR.name}\n')

    failures = []
    pads = []

    for eng in engagements:
        name = eng.source_path.name
        pts = np.vstack([t.points for t in eng.tracks])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        box, ticks = _nice_box(lo, hi, N_LABELS)

        for axis, label in enumerate('XYZ'):
            a, b, n = box[2 * axis], box[2 * axis + 1], ticks[axis]

            if a > lo[axis] or b < hi[axis]:
                failures.append(
                    f'{name} {label}: box [{a:.0f}, {b:.0f}] crops data '
                    f'[{lo[axis]:.0f}, {hi[axis]:.0f}]')

            # Ticks are round when every one is an integer multiple of the step.
            step = (b - a) / (n - 1)
            values = np.linspace(a, b, n)
            off = np.abs(values / step - np.round(values / step))
            if off.max() > 1e-9:
                worst = values[int(np.argmax(off))]
                failures.append(
                    f'{name} {label}: tick {worst:.3f} is not a multiple '
                    f'of step {step:.3f}')

            span = hi[axis] - lo[axis]
            pads.append((b - a) - span if span > 0 else 0.0)

        for row in eng.verify():
            if not row['ok']:
                failures.append(
                    f'{name} {row["label"]}: min_R recorded={row["recorded"]} '
                    f'recomputed={row["recomputed"]}')

    pads = np.array(pads)
    print(f'padding per axis: mean {pads.mean():.0f} m, '
          f'max {pads.max():.0f} m, min {pads.min():.0f} m')

    if failures:
        print(f'\n{len(failures)} FAILURES:')
        for f in failures:
            print('  ', f)
        return 1

    print(f'\nall {len(engagements)} episodes pass: no cropping, '
          f'round ticks, min_R exact')
    return 0


if __name__ == '__main__':
    sys.exit(main())
