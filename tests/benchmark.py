"""Load + render time benchmark for pytrajectory.

Measures how long the program takes to turn a trajectory into output, and how
that grows with the number of points and the number of trajectories. Reports a
table (and writes benchmark_results.csv).

Run from the repo root:
    python tests/benchmark.py
"""
import csv
import os
import sys
import time

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

from pytrajectory.core import flypath3d, flypath3d_multi
from pytrajectory.io import load_trajectory


def make_helix(n):
    """A synthetic n-point trajectory (x, y, z)."""
    t = np.linspace(0, 8 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), t / (2 * np.pi)])


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench_static(points, repeats=3):
    """Time a full off-screen render (build scene + one screenshot)."""
    best = float("inf")
    for _ in range(repeats):
        def run():
            p = flypath3d(points, off_screen=True, return_plotter=True)
            p.screenshot(return_img=True)
            p.close()
        best = min(best, timed(run))
    return best


def bench_gif(points, out):
    """Time an animated GIF export (90 frames, sphere marker)."""
    return timed(lambda: flypath3d(points, animate=True, off_screen=True,
                                   save_animation=out))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    sizes = [60, 200, 1000, 5000, 20000, 60000]

    # One warmup so VTK/GL init overhead isn't charged to the first row.
    p = flypath3d(make_helix(60), off_screen=True, return_plotter=True)
    p.screenshot(return_img=True); p.close()

    rows = []
    print("\n=== Single trajectory: load + render vs size ===")
    print(f"{'points':>8} | {'load (s)':>9} | {'render (s)':>11} | {'gif 90f (s)':>12}")
    print("-" * 50)
    for n in sizes:
        pts = make_helix(n)
        # write a temp csv so load time reflects real file parsing
        csv_path = os.path.join(out_dir, f"_bench_{n}.csv")
        np.savetxt(csv_path, pts, delimiter=",", header="x,y,z", comments="")
        t_load = min(timed(lambda: load_trajectory(csv_path)) for _ in range(3))
        t_render = bench_static(pts)
        t_gif = bench_gif(pts, os.path.join(out_dir, "_bench.gif"))
        os.remove(csv_path)
        print(f"{n:>8} | {t_load:>9.3f} | {t_render:>11.3f} | {t_gif:>12.3f}")
        rows.append({"kind": "single", "points": n, "trajectories": 1,
                     "load_s": round(t_load, 4), "render_s": round(t_render, 4),
                     "gif_s": round(t_gif, 4)})

    print("\n=== Multiple trajectories: render vs count (2000 pts each) ===")
    print(f"{'trajs':>8} | {'render (s)':>11}")
    print("-" * 24)
    base = make_helix(2000)
    for k in (1, 2, 3, 5):
        trajs = [{"data": base + i * 0.1, "color": "blue"} for i in range(k)]
        def run():
            p = flypath3d_multi(trajs, off_screen=True, return_plotter=True)
            p.screenshot(return_img=True); p.close()
        t = min(timed(run) for _ in range(3))
        print(f"{k:>8} | {t:>11.3f}")
        rows.append({"kind": "multi", "points": 2000, "trajectories": k,
                     "load_s": "", "render_s": round(t, 4), "gif_s": ""})

    print("\n=== Real bundled files (static render) ===")
    for f in ["examples/trajectory_aircraft.mat", "examples/trajectory_aam.mat",
              "examples/sample_data.csv"]:
        if not os.path.exists(f):
            continue
        pts, _, _ = load_trajectory(f)
        t = bench_static(pts)
        print(f"{f:<40} {len(pts):>7} pts  {t:.3f}s")

    csv_out = os.path.join(out_dir, "benchmark_results.csv")
    with open(csv_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "points", "trajectories",
                                           "load_s", "render_s", "gif_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_out}")


if __name__ == "__main__":
    main()
