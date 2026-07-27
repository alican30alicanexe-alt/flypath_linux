# pytrajectory

A Python clone of MATLAB's `flypath3d` for precision 3D trajectory visualization.

Uses **PyVista (VTK)** for high-quality, MATLAB-like 3D rendering with equal aspect ratio, proper grids, and real-scale coordinate preservation.

## Features

- **Single trajectory** plotting from CSV or numpy arrays
- **Multi-trajectory** overlay in the same 3D scene
- **3D model animation** — attach F-16, missile, or custom models to trajectories
- **Multi-model animation** — multiple models following different paths simultaneously
- **Orientation support** — pitch, yaw, roll from trajectory data, auto-aligned to the flight path
- **Flight trail** — reveal the path progressively as the model flies over it
- **Vertical exaggeration** — stretch a flat/altitude axis without distorting the tick values
- **Top-down / side presets** — flat orthographic scene views
- **Animation export** — save as GIF
- **Standalone executable** — build with PyInstaller (no Python required)

## Installation

```bash
# From source
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

## Usage

### Single trajectory from CSV

```bash
pytrajectory examples/sample_data.csv --color blue
pytrajectory examples/sample_data.csv --colormap viridis --title "My Trajectory"
```

### With animation and 3D model

```bash
pytrajectory examples/sample_data.csv --animate --model models/f-16.mat

# Reveal the path as a growing flight trail, and show start/end markers
pytrajectory examples/trajectory_aircraft.mat --animate --model models/f-16.mat \
  --trail --markers
```

### Display options

```bash
# Line width uses a fine scale: default (15) is a thin line, and 100 ~= the
# classic thickness. Use small values for hairlines:
pytrajectory trajectory.csv --line-width 100
pytrajectory trajectory.csv --line-width 5

# Model size (alias for --model-scale):
pytrajectory trajectory.csv --model models/f-16.mat --scale 10 --animate

# Markers (green start / red end) are hidden by default — show them:
pytrajectory trajectory.csv --markers

# Vertical exaggeration for flat trajectories (z-range << x/y-range).
# Tick labels keep real values; only the geometry is stretched:
pytrajectory trajectory.csv --z-scale 8

# Manual axis limits (also reframe the camera, not just the grid box):
pytrajectory trajectory.csv --xlim -500 500 --ylim -500 3500 --zlim 500 1500
```

### Orientation / angle conventions

The trajectory attitude columns default to the **aerospace** convention
(yaw = compass/clockwise-positive, pitch = nose-up-positive), matching typical
flight `.mat` data. The model is auto-mounted onto the path's initial heading,
so it stays aligned through turns.

```bash
# Switch to the math convention (yaw counter-clockwise-positive):
pytrajectory trajectory.csv --animate --model models/f-16.mat --math

# Or invert a single axis if the model turns/pitches/banks the wrong way:
pytrajectory trajectory.csv --animate --model models/f-16.mat --flip-yaw
```

### Multiple trajectories

Each trajectory is its own `--traj` block: a file path followed by `key=value`
options. Global `--color`/`--colormap`/`--model`/`--scale`/`--line-width` act as
defaults, and per-block keys override them.

```bash
pytrajectory \
  --traj traj1.csv color=red  label=Missile \
  --traj traj2.csv color=blue label=Aircraft
```

Per-block keys: `color`, `colormap`, `label`, `model` (path to a `.mat` model),
`mcolor` (model color, default gray), `scale` (model size), `lw` (line width),
`face` (`data` = use attitude columns, default; `path` = align to direction of
travel — use for long projectiles like a scaled-up missile whose attitude data
would otherwise make them slalom off the path).

```bash
pytrajectory \
  --traj missile.csv color=red  label=Missile model=models/missile.mat scale=10 \
  --traj target.csv  color=blue label=Target  model=models/f-16.mat   scale=10 mcolor=green \
  --animate --trail
```

Colors accept names, hex (`#ff0000`), or `r,g,b` floats (e.g. `color=0.9,0.3,0`).

### Demos

```bash
pytrajectory --demo spiral
pytrajectory --demo helix
pytrajectory --demo figure8
pytrajectory --demo multi        # Multiple trajectories together
pytrajectory --demo all          # Run all demos
pytrajectory --list-demos
```

### Python API

```python
from pytrajectory import flypath3d, flypath3d_multi
import numpy as np

# Single trajectory
points = np.column_stack([x, y, z])
flypath3d(points, title='My Trajectory', colormap='viridis')

# Multiple trajectories
trajectories = [
    {'data': 'missile.csv', 'color': 'red', 'label': 'Missile'},
    {'data': 'target.csv', 'color': 'blue', 'label': 'Target'},
]
models = [
    {'path': 'models/missile.mat', 'trajectory_index': 0, 'color': 'gray'},
]
flypath3d_multi(trajectories, models=models, title='Engagement', animate=True)
```

### Combat scene example

`examples/combat_example.py` reproduces the classic flypath3d combat demo — a
friendly aircraft, an enemy aircraft, and an air-to-air missile, each with its
own path color and model, viewed top-down. It writes a static multi-exposure
PNG and an animated GIF.

```bash
python examples/combat_example.py
```

It uses these `flypath3d_multi` options:

```python
flypath3d_multi(
    trajectories,               # each: {'data', 'color'(name/hex/RGB tuple)}
    models=[                    # per model:
        {'path': 'models/f-16.mat', 'trajectory_index': 0,
         'color': (.3, .3, .3),
         'scale': 1.0,          # size, as a fraction of the scene
         'count': 6},           # static copies placed along the path (PNG only)
    ],
    view='top',                 # 'top' | 'side' | 'iso'  (orthographic for top/side)
    window_size=(1000, 280),    # pixels
    xlim=(-500, 500), ylim=(-500, 3500), zlim=(500, 1500),
)
```

## Building Standalone Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Build
chmod +x build.sh
./build.sh

# Or manually:
pyinstaller pytrajectory.spec --clean

# Run the binary
./dist/pytrajectory/pytrajectory --demo spiral
```

The standalone binary includes all models and example data, and runs on any Linux system without Python installed.

## Data Format

### CSV files
```
x, y, z, pitch, yaw, roll
0.0, 0.0, 0.0, 0, 0, 0
...
```
Columns 0-2: x, y, z coordinates
Columns 3-5: pitch, yaw, roll angles (optional). CSV angles are **degrees** by
default (pass `--rad` for radians); `.mat` files are assumed to be radians.

### 3D Model files (.mat)
MATLAB .mat files containing:
- `V`: (N, 3) vertex coordinates
- `F`: (M, 3) face indices (triangles)
- `C`: (N, 1) color indices (optional)

## License

MIT