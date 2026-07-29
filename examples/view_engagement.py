"""Open an engagement export in an interactive 3D window.

    python examples/view_engagement.py [path/to/episode.npz]

With no argument it picks the first episode in the most recent export
directory. Drag to orbit, scroll to zoom, 'q' to close.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pytrajectory.engagement import load_engagement, render_engagement

EVAL_ROOT = Path('/home/mmsgb/Desktop/TDTP/TDTP_13Jul/3DOF_v3/EvalTrajectories')


def newest_episode():
    """First episode of the most recently written export run."""
    runs = sorted((d for d in EVAL_ROOT.iterdir() if d.is_dir()), reverse=True)
    for run in runs:
        episodes = sorted(run.glob('*.npz'))
        if episodes:
            return episodes[0]
    raise FileNotFoundError(f'No .npz exports found under {EVAL_ROOT}')


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_episode()
    eng = load_engagement(path)

    print(f'{path.name}: {eng.title()}')
    print(f'  case   : {eng.case}')
    print(f'  source : {eng.meta.get("source", "?")}')
    for track in eng.tracks:
        print(f'  {track.label:8s} {len(track):5d} samples  {track.color}')

    # Confirm the picture agrees with the recorded numbers before showing it.
    for row in eng.verify():
        mark = 'ok' if row['ok'] else 'MISMATCH'
        print(f'  verify {row["label"]:8s} min_R recorded={row["recorded"]:.3f} '
              f'recomputed={row["recomputed"]:.3f}  [{mark}]')

    plotter = render_engagement(eng, off_screen=False)
    # auto_close=False keeps the render window up after the interactor returns;
    # without it the window can close the moment it finishes its first render.
    plotter.show(title=path.name, interactive=True, auto_close=False)
    plotter.close()


if __name__ == '__main__':
    main()
