"""Generate controlled orientation test trajectories.

Each is a STRAIGHT line along +X (so the base heading alignment is identity),
with exactly one of yaw/pitch/roll swept through a known range. This isolates
each Euler column so we can see whether the model rotates the right amount, in
the right direction, about the right axis.

Angles are in DEGREES (CSV default; no --rad needed).
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / 'orientation'
OUT.mkdir(exist_ok=True)

N = 60
x = np.linspace(0, 100, N)
y = np.zeros(N)
z = np.full(N, 50.0)
zeros = np.zeros(N)


def write(name, pitch, yaw, roll):
    df = pd.DataFrame({'x': x, 'y': y, 'z': z,
                       'pitch': pitch, 'yaw': yaw, 'roll': roll})
    df.to_csv(OUT / name, index=False)
    print(f"wrote {name}: yaw {yaw[0]:.0f}->{yaw[-1]:.0f}  "
          f"pitch {pitch[0]:.0f}->{pitch[-1]:.0f}  roll {roll[0]:.0f}->{roll[-1]:.0f}")


# Pure yaw: nose should swing from +X toward +Y (left) by 90 deg
write('yaw_0_90.csv', zeros, np.linspace(0, 90, N), zeros)
# Pure pitch: nose should tilt up by 45 deg
write('pitch_0_45.csv', np.linspace(0, 45, N), zeros, zeros)
# Pure roll: model should bank about its nose axis by 90 deg
write('roll_0_90.csv', zeros, zeros, np.linspace(0, 90, N))
# Diagonal path, zero angles: model should simply point along the diagonal
xd = np.linspace(0, 100, N)
df = pd.DataFrame({'x': xd, 'y': xd, 'z': np.full(N, 50.0),
                   'pitch': zeros, 'yaw': zeros, 'roll': zeros})
df.to_csv(OUT / 'diagonal_zero.csv', index=False)
print("wrote diagonal_zero.csv: path +X+Y, all angles 0")
