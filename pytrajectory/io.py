"""
File loading utilities for pytrajectory.
Supports CSV and MATLAB .mat files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat


def load_csv(filepath, x_col=None, y_col=None, z_col=None,
             pitch_col=None, yaw_col=None, roll_col=None):
    """Load trajectory data from a CSV file.
    
    Automatically detects x, y, z columns by name (case-insensitive),
    looking for common patterns like 'x', 'X', 'pos_x', 'x_pos', etc.
    Also auto-detects pitch, yaw, roll columns if present.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    x_col : str, optional
        Name of the x-coordinate column. Auto-detected if not provided.
    y_col : str, optional
        Name of the y-coordinate column. Auto-detected if not provided.
    z_col : str, optional
        Name of the z-coordinate column. Auto-detected if not provided.
    pitch_col : str, optional
        Name of the pitch column. Auto-detected if not provided.
    yaw_col : str, optional
        Name of the yaw column. Auto-detected if not provided.
    roll_col : str, optional
        Name of the roll column. Auto-detected if not provided.
    
    Returns
    -------
    tuple of (np.ndarray, pd.DataFrame, dict)
        points : (N, 3) array of [x, y, z] coordinates
        dataframe : full pandas DataFrame with all columns
        col_map : dict with keys 'pitch', 'yaw', 'roll' mapping to column names (or None)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Auto-detect columns if not specified
    if x_col is None:
        x_col = _find_column(df, ['x', 'pos_x', 'x_pos', 'position_x', 'x_coord', 'x_coordinate'])
    if y_col is None:
        y_col = _find_column(df, ['y', 'pos_y', 'y_pos', 'position_y', 'y_coord', 'y_coordinate'])
    if z_col is None:
        z_col = _find_column(df, ['z', 'pos_z', 'z_pos', 'position_z', 'z_coord', 'z_coordinate'])
    
    if x_col is None or y_col is None or z_col is None:
        raise ValueError(
            "Could not auto-detect x, y, z columns. "
            "Please specify them manually with x_col, y_col, z_col parameters. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Auto-detect orientation columns
    if pitch_col is None:
        pitch_col = _find_column(df, ['pitch', 'theta', 'pitch_angle', 'elevation'])
    if yaw_col is None:
        yaw_col = _find_column(df, ['yaw', 'psi', 'yaw_angle', 'heading', 'azimuth'])
    if roll_col is None:
        roll_col = _find_column(df, ['roll', 'phi', 'roll_angle', 'bank'])
    
    col_map = {'pitch': pitch_col, 'yaw': yaw_col, 'roll': roll_col}
    
    points = np.column_stack([df[x_col].values, df[y_col].values, df[z_col].values])
    return points, df, col_map


def load_mat(filepath, x_key='x', y_key='y', z_key='z'):
    """Load trajectory data from a MATLAB .mat file.
    
    If x/y/z variable names are not found, falls back to using the
    first 3 columns of the first 2D array variable in the file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .mat file.
    x_key : str, default 'x'
        Variable name for x-coordinates in the .mat file.
    y_key : str, default 'y'
        Variable name for y-coordinates in the .mat file.
    z_key : str, default 'z'
        Variable name for z-coordinates in the .mat file.
    
    Returns
    -------
    np.ndarray : (N, 3) array of [x, y, z] coordinates
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MAT file not found: {filepath}")
    
    mat = loadmat(filepath)
    available_keys = [k for k in mat.keys() if not k.startswith('_')]
    
    # Try named variables first
    x_key_found = x_key if x_key in mat else None
    y_key_found = y_key if y_key in mat else None
    z_key_found = z_key if z_key in mat else None
    
    if x_key_found is None:
        for k in available_keys:
            if 'x' in k.lower():
                x_key_found = k
                break
    if y_key_found is None:
        for k in available_keys:
            if 'y' in k.lower() and k != x_key_found:
                y_key_found = k
                break
    if z_key_found is None:
        for k in available_keys:
            if 'z' in k.lower() and k not in (x_key_found, y_key_found):
                z_key_found = k
                break
    
    if x_key_found and y_key_found and z_key_found:
        x = np.asarray(mat[x_key_found]).flatten()
        y = np.asarray(mat[y_key_found]).flatten()
        z = np.asarray(mat[z_key_found]).flatten()
        min_len = min(len(x), len(y), len(z))
        return np.column_stack([x[:min_len], y[:min_len], z[:min_len]])
    else:
        # Fallback: use the first 2D array variable (may have 3+ columns)
        arr_var = None
        for k in available_keys:
            v = mat[k]
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= 3:
                arr_var = v
                break
        if arr_var is None:
            raise ValueError(
                f"Could not find x, y, z arrays or a suitable 2D matrix in .mat file. "
                f"Available variables: {available_keys}"
            )
        return arr_var


def load_trajectory(data):
    """Load trajectory data from CSV, .mat file, or numpy array.
    
    Parameters
    ----------
    data : str, Path, or np.ndarray
        Path to CSV or .mat file, or (N, 3) array of x,y,z points.
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None, dict)
        points : (N, 3) array of [x, y, z] coordinates
        full_data : (N, M) array with all columns, or None if numpy input
        info : dict with keys:
            - 'col_map': dict mapping 'pitch'/'yaw'/'roll' to column indices (or None)
            - 'from_mat': bool, True if loaded from .mat file
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() == '.mat':
            full_data = load_mat(path)
            # First 3 columns are x, y, z; rest may be pitch, yaw, roll
            points = full_data[:, :3]
            # Only set orientation columns if data has enough columns
            if full_data.shape[1] >= 6:
                col_map = {'pitch': 3, 'yaw': 4, 'roll': 5}
            else:
                col_map = {}
            return points, full_data, {'col_map': col_map, 'from_mat': True}
        
        points, df, col_map = load_csv(path)
        # Map detected column names to indices
        col_indices = {}
        for key, col_name in col_map.items():
            if col_name is not None and col_name in df.columns:
                col_indices[key] = list(df.columns).index(col_name)
        return points, df.values, {'col_map': col_indices, 'from_mat': False}
    
    arr = np.asarray(data, dtype=float)
    return arr, None, {'col_map': {}, 'from_mat': False}


def load_3d_model(filepath):
    """Load 3D mesh model from a MATLAB .mat file.
    
    Expects .mat file to contain:
    - V: (N, 3) array of vertex coordinates
    - F: (M, 3) array of face indices (triangles)
    - C: (N, 1) array of color indices (optional)
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .mat file containing 3D model data.
    
    Returns
    -------
    pyvista.PolyData
        PyVista mesh object with vertices and faces.
    """
    import pyvista as pv
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    mat = loadmat(filepath)
    available_keys = [k for k in mat.keys() if not k.startswith('_')]
    
    # Find vertices (V)
    if 'V' in mat:
        vertices = np.asarray(mat['V'], dtype=np.float64)
    else:
        raise ValueError(
            f"Could not find 'V' (vertices) array in .mat file. "
            f"Available variables: {available_keys}"
        )
    
    # Find faces (F)
    if 'F' in mat:
        faces = np.asarray(mat['F'], dtype=np.int64)
    else:
        raise ValueError(
            f"Could not find 'F' (faces) array in .mat file. "
            f"Available variables: {available_keys}"
        )
    
    # Validate shapes
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Vertices must be (N, 3) array, got shape {vertices.shape}")
    
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Faces must be (M, 3) array, got shape {faces.shape}")
    
    # Create PyVista mesh
    # MATLAB .mat files use 1-based indexing, VTK uses 0-based
    # PyVista expects faces in format: [n_points, v1, v2, v3, n_points, v1, v2, v3, ...]
    # where n_points is 3 for triangles
    n_faces = len(faces)
    faces_0based = faces - 1  # Convert from MATLAB 1-based to VTK 0-based
    faces_pv = np.column_stack([np.full(n_faces, 3), faces_0based]).flatten()
    
    mesh = pv.PolyData(vertices, faces_pv)
    
    # Add color data if available
    if 'C' in mat:
        colors = np.asarray(mat['C']).flatten()
        if len(colors) == len(vertices):
            mesh.point_data['colors'] = colors
    
    return mesh


def _find_column(df, candidates):
    """Find a column in DataFrame by trying candidate names (case-insensitive)."""
    col_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in col_lower:
            return col_lower[candidate]
    return None
