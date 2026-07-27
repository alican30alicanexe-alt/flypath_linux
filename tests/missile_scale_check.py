"""Render trajectory_aam with missile.mat at scale 1 vs scale 10, saving a
screenshot every 1/10 of the animation for each, to inspect whether the missile
tracks the trail cleanly or jumps around as it is scaled up.

Usage: python tests/missile_scale_check.py <out_dir>
"""
import sys
import types
import numpy as np
from PIL import Image

# Pillow-based imageio shim so we can capture raw frames regardless of version.
_mod = types.ModuleType('imageio')
_cap = {}
_mod.mimsave = lambda p, f, fps=30, loop=0: _cap.__setitem__('f', f)
sys.modules['imageio'] = _mod

import pyvista as pv
pv.OFF_SCREEN = True
from pytrajectory.core import flypath3d_multi

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'

for scale in (1.0, 10.0):
    trajectories = [{'data': 'examples/trajectory_aam.mat', 'color': 'orange'}]
    models = [{'path': 'models/missile.mat', 'trajectory_index': 0,
               'color': 'gray', 'scale': scale}]
    flypath3d_multi(trajectories, models=models, off_screen=True,
                    animate=True, trail=True, view='top',
                    save_animation=f'{OUT}/aam_s{int(scale)}.gif')
    frames = _cap['f']
    n = len(frames)
    for i in range(10):
        idx = min(n - 1, int(round(i / 10 * n)))
        Image.fromarray(frames[idx]).save(f'{OUT}/aam_s{int(scale)}_{i}.png')
    print(f'scale {scale}: {n} frames, wrote aam_s{int(scale)}_0..9.png')
