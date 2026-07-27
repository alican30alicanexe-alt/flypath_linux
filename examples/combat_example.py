"""Combat scene example — a Python port of the flypath3d MATLAB demo.

Three trajectories (friendly aircraft, enemy aircraft, air-to-air missile) are
drawn with their own path colors and 3D models, viewed top-down. Produces both
a static multi-exposure PNG (several model copies along each path) and an
animated GIF (one model flying each path).

Run from the repo root:
    python examples/combat_example.py
"""
import os
from pytrajectory.core import flypath3d_multi

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

# Path colors matching the MATLAB demo
GRAY = (0.30, 0.30, 0.30)   # enemy
RED = (0.89, 0.27, 0.00)    # friendly
ORANGE = (0.89, 0.47, 0.00)  # missile

trajectories = [
    {"data": "examples/trajectory_friend.mat", "color": RED},
    {"data": "examples/trajectory_enemy.mat", "color": GRAY},
    # The missile is long/thin, so face it along the path to avoid slalom.
    {"data": "examples/trajectory_aam.mat", "color": ORANGE, "face": "path"},
]

# One model per trajectory. `scale` is in our convention (fraction of the scene
# size); `count` places that many static copies along the path for the PNG.
LIMITS = dict(xlim=(-500, 500), ylim=(-500, 3500), zlim=(500, 1500))
VIEW = dict(view="top", window_size=(1000, 280))

# --- Static multi-exposure PNG ---
static_models = [
    {"path": "models/f-16.mat", "trajectory_index": 0, "color": GRAY, "count": 6, "scale": 1.0},
    {"path": "models/f-16.mat", "trajectory_index": 1, "color": GRAY, "count": 6, "scale": 1.0},
    {"path": "models/missile.mat", "trajectory_index": 2, "color": GRAY, "count": 6, "scale": 1.0},
]
p = flypath3d_multi(trajectories, models=static_models, off_screen=True,
                    return_plotter=True, **VIEW, **LIMITS)
p.screenshot("combat_example.png", scale=3)
print("wrote combat_example.png")

# --- Animated GIF (one model flying each path) ---
anim_models = [
    {"path": "models/f-16.mat", "trajectory_index": 0, "color": GRAY, "scale": 1.5},
    {"path": "models/f-16.mat", "trajectory_index": 1, "color": GRAY, "scale": 1.5},
    {"path": "models/missile.mat", "trajectory_index": 2, "color": GRAY, "scale": 1.5},
]
flypath3d_multi(trajectories, models=anim_models, off_screen=True,
                animate=True, save_animation="combat_example.gif",
                trail=True, speed=3.0, **VIEW, **LIMITS)
print("wrote combat_example.gif")
