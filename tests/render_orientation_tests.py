"""Render start/mid/end frames of each orientation test so the model's
rotation can be read off directly (default iso camera)."""
import sys, types
from PIL import Image

# Pillow-based imageio shim (independent of imageio version)
mod = types.ModuleType('imageio'); cap = {}
mod.mimsave = lambda p, f, fps=30, loop=0: cap.__setitem__('f', f)
sys.modules['imageio'] = mod

import pyvista as pv
pv.OFF_SCREEN = True
from pytrajectory.core import flypath3d

OUT = sys.argv[1]
TESTS = ['yaw_0_90', 'pitch_0_45', 'roll_0_90', 'diagonal_zero']

for t in TESTS:
    flypath3d(f'tests/orientation/{t}.csv', model='models/f-16.mat',
              model_scale=8.0, animate=True,
              save_animation=f'{OUT}/{t}.gif', off_screen=True)
    f = cap['f']; n = len(f)
    for tag, i in [('start', 0), ('mid', n // 2), ('end', n - 1)]:
        Image.fromarray(f[i]).save(f'{OUT}/{t}_{tag}.png')
    print(f'{t}: {n} frames captured')
