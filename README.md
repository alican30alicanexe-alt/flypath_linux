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

## Showcase

A taste of what pytrajectory can do beyond a single trajectory — three paths,
their own 3D models, colors, a thick trail, and a flying animation, all from
one command. Friendly and enemy aircraft plus a missile, each with its own
model and color:

```bash
# pip install: run as "pytrajectory". Standalone build: run from inside the
# extracted folder as "./pytrajectory" instead.
pytrajectory \
  --traj examples/trajectory_friend.mat color=green model=models/f-16.mat \
  --traj examples/trajectory_enemy.mat  color=black model=models/f-16.mat \
  --traj examples/trajectory_aam.mat    color=red   model=models/missile.mat scale=10 \
  --line-width 100 --animate --trail \
  --xlim -3000 1000 --zlim 0 1000
```

Windows (PowerShell) — same command, backtick line continuation and backslash paths:

```powershell
# pip install: run as "pytrajectory". Standalone build: PowerShell won't run
# an exe from the current folder by bare name, run it from inside the
# extracted folder as ".\pytrajectory.exe" instead.
pytrajectory `
  --traj examples\trajectory_friend.mat color=green model=models\f-16.mat `
  --traj examples\trajectory_enemy.mat  color=black model=models\f-16.mat `
  --traj examples\trajectory_aam.mat    color=red   model=models\missile.mat scale=10 `
  --line-width 100 --animate --trail `
  --xlim -3000 1000 --zlim 0 1000
```

What each part does:

- `--traj <path> key=value ...` — one block per trajectory; repeat it for as
  many paths as you want in the same scene (here: friend, enemy, missile).
- `color=` — path/trail color for that trajectory.
- `model=` — attaches a 3D model (`.mat` file) that flies along the path.
- `scale=10` — the missile model is scaled 10x. Missiles are much smaller
  than aircraft in real proportions, so without this it's barely visible.
- `--line-width 100` — a thick trail line (default is 50).
- `--animate --trail` — flies each model along its path and reveals the
  trail progressively as the model passes over it, instead of drawing the
  whole path upfront.
- `--xlim ... --zlim ...` — pins the camera box to a fixed size instead of
  auto-fitting to the data.

Drop `--xlim`/`--zlim` and the box auto-fits to the data instead — a bigger,
uncropped view, handy when you don't already know the scene's extent. An
auto-fitted axis is snapped *outward* onto round numbers, so a track spanning
-323..2734 is drawn as -500..3000 and gridded in 500s rather than at whatever
values a percentage pad landed on:

```bash
# pip install: "pytrajectory". Standalone build: "./pytrajectory".
pytrajectory \
  --traj examples/trajectory_friend.mat color=green model=models/f-16.mat \
  --traj examples/trajectory_enemy.mat  color=black model=models/f-16.mat \
  --traj examples/trajectory_aam.mat    color=red   model=models/missile.mat scale=10 \
  --line-width 100 --animate --trail
```

Windows (PowerShell):

```powershell
# pip install: "pytrajectory". Standalone build: ".\pytrajectory.exe".
pytrajectory `
  --traj examples\trajectory_friend.mat color=green model=models\f-16.mat `
  --traj examples\trajectory_enemy.mat  color=black model=models\f-16.mat `
  --traj examples\trajectory_aam.mat    color=red   model=models\missile.mat scale=10 `
  --line-width 100 --animate --trail
```

> Tip: a long, thin model (like a scaled-up missile) can wobble/slalom if its
> attitude data is noisy. Add `face=path` to that trajectory's block to point
> it along the direction of travel instead of using its attitude columns.

Both commands are also in [`examples/cli_showcase.sh`](examples/cli_showcase.sh)
(Linux/macOS) and [`examples/cli_showcase.ps1`](examples/cli_showcase.ps1)
(Windows) if you'd rather copy from a file than the README.

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
# Line width uses a fine scale: default (50) is a moderate line, and 100 ~=
# the classic thickness. Use small values for hairlines:
pytrajectory trajectory.csv --line-width 100
pytrajectory trajectory.csv --line-width 5

# Model size (alias for --model-scale):
pytrajectory trajectory.csv --model models/f-16.mat --scale 10 --animate

# Markers (green start / red end) are hidden by default — show them:
pytrajectory trajectory.csv --markers

# Vertical exaggeration for flat trajectories (z-range << x/y-range).
# Tick labels keep real values; only the geometry is stretched:
pytrajectory trajectory.csv --z-scale 8

# Manual axis limits (also reframe the camera, not just the grid box).
# Given limits are used exactly as typed — never widened to reach a round
# number — while any axis you leave out still snaps to one on its own:
pytrajectory trajectory.csv --xlim -500 500 --ylim -500 3500 --zlim 500 1500
```

### Axis limits and grid steps

Axes you don't pin are fitted automatically, and the fit always lands on round
numbers — ticks step in 1s, 2s, 2.5s or 5s times a power of ten (250 / 500 /
1000 …), never on the raw data extent divided into equal parts. The box only
ever grows, so no sample is ever cropped, and each axis is snapped on its own,
which leaves the true 1:1:1 aspect untouched.

Two details worth knowing:

- **Label density follows axis length.** Under a real 1:1:1 aspect a short axis
  is short on screen, so it is given proportionally fewer ticks — otherwise its
  labels overlap into an unreadable smear.
- **A too-thin axis is given a box of its own.** An axis spanning less than one
  grid step of the scene — a constant altitude, a planar path, a track that
  wanders 76 m over 3 km — renders as a pancake under a true 1:1:1 aspect, and a
  perfectly flat one has no extent to snap to at all. Both get a box two grid
  steps wide, centred on the data, so the result follows how thin the axis is
  rather than where it happens to sit: in a scene gridded in 500s a track at
  z=0 spans -500..500 and one at z=600 spans 0..1000. Note this only buys the
  box some depth — it cannot reveal detail that is genuinely flat at 1:1:1. Use
  `--z-scale` for that, or pin the axis with `--zlim`.

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

## Standalone Executable (no Python required)

### Download a prebuilt binary

Linux and Windows builds are published on the
[Releases page](https://github.com/alican30alicanexe-alt/flypath_linux/releases) —
download, extract, and run. No Python, no `pip install`. See the release notes
there for platform-specific quick-start commands.

### Or build it yourself

Linux/macOS:

```bash
pip install pyinstaller pyinstaller-hooks-contrib

chmod +x build.sh
./build.sh

# Run it
./dist/pytrajectory/pytrajectory --demo spiral
```

Windows (PowerShell):

```powershell
pip install -e .
pip install pyinstaller pyinstaller-hooks-contrib

pyinstaller pytrajectory.spec --clean
Copy-Item -Recurse models dist\pytrajectory\models
Copy-Item -Recurse examples dist\pytrajectory\examples

# Run it
.\dist\pytrajectory\pytrajectory.exe --demo spiral
```

Either way, `dist/pytrajectory/` (or `dist\pytrajectory\` on Windows) ends up
self-contained: the executable, the Python/PyVista/VTK runtime, and the
`models/`/`examples/` folders all sit side by side, so the whole folder can be
zipped and moved to another machine without Python installed. `pytrajectory.spec`
does *not* bundle `models/`/`examples/` itself — `build.sh` (and the
[`build-release`](.github/workflows/build-release.yml) CI workflow) copy them
in as a separate step after PyInstaller runs, since PyInstaller's own
data-bundling puts them somewhere the executable can't find by relative path.

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