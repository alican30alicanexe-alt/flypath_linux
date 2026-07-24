"""
pytrajectory - A Python clone of MATLAB's flypath3d for precision 3D trajectory visualization.

Uses PyVista (VTK) for high-quality, MATLAB-like 3D rendering with
equal aspect ratio, proper grids, and real-scale coordinate preservation.
"""

from .core import flypath3d, flypath3d_multi, plot_trajectory
from .io import load_csv, load_trajectory

__version__ = "0.1.0"
__all__ = ["flypath3d", "flypath3d_multi", "plot_trajectory", "load_csv", "load_trajectory"]
