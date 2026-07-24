# pytrajectory

A Python clone of MATLAB's `flypath3d` for precision 3D trajectory visualization.

Uses **PyVista (VTK)** for high-quality, MATLAB-like 3D rendering with equal aspect ratio, proper grids, and real-scale coordinate preservation.

## Features

- **Single trajectory** plotting from CSV or numpy arrays
- **Multi-trajectory** overlay in the same 3D scene
- **3D model animation** — attach F-16, missile, or custom models to trajectories
- **Multi-model animation** — multiple models following different paths simultaneously
- **Orientation support** — pitch, yaw, roll from trajectory data
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
```

### Multiple trajectories

```bash
pytrajectory --multi traj1.csv:red:Missile traj2.csv:blue:Aircraft
```

### Multiple trajectories with 3D models

```bash
pytrajectory --multi missile.csv:red:Missile target.csv:blue:Target \
  --models models/missile.mat:0:gray models/f-16.mat:1:green
```

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
Columns 3-5: pitch, yaw, roll angles (radians, optional)

### 3D Model files (.mat)
MATLAB .mat files containing:
- `V`: (N, 3) vertex coordinates
- `F`: (M, 3) face indices (triangles)
- `C`: (N, 1) color indices (optional)

## License

MIT