"""
Core plotting functions for pytrajectory.
MATLAB-like 3D trajectory visualization using PyVista.
"""

import numpy as np
import pyvista as pv
from pathlib import Path
from .io import load_trajectory, load_3d_model


def _euler_to_matrix(pitch_deg, yaw_deg, roll_deg):
    """Vectorized rotation matrices for arrays of Euler angles (degrees).

    Returns (N, 3, 3) matrices R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching the
    standard aerospace/MATLAB convention. VTK's actor.SetOrientation applies a
    different composition (Rz @ Rx @ Ry), so we build the matrix explicitly
    instead of relying on it.
    """
    p = np.radians(np.asarray(pitch_deg, dtype=float))
    y = np.radians(np.asarray(yaw_deg, dtype=float))
    r = np.radians(np.asarray(roll_deg, dtype=float))
    n = p.shape[0]

    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)

    Rz = np.zeros((n, 3, 3))
    Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz
    Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz
    Rz[:, 2, 2] = 1.0

    Ry = np.zeros((n, 3, 3))
    Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy

    Rx = np.zeros((n, 3, 3))
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cx; Rx[:, 1, 2] = -sx
    Rx[:, 2, 1] = sx; Rx[:, 2, 2] = cx

    return Rz @ Ry @ Rx


def _alignment_rotation(direction):
    """Rotation matrix that turns the model's nose (+X body axis) to point along
    `direction`, keeping world +Z as "up" (yaw + pitch only, no roll).

    Models here follow the aerospace body convention: nose along +X (the
    longitudinal/roll axis), which matches the provided meshes (f-16, missile
    are longest along X) and the roll/pitch/yaw columns in the trajectory data.
    """
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.eye(3)
    d = d / norm
    horiz = np.hypot(d[0], d[1])
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(-d[2], horiz)  # matches Ry sign in _euler_to_matrix
    return _euler_to_matrix([np.degrees(pitch)], [np.degrees(yaw)], [0.0])[0]


def _trajectory_tangents(points, indices):
    """Unit direction of travel at each of the given point indices (via central
    differences), for aligning a model that has no attitude data."""
    diffs = np.gradient(points, axis=0)
    tang = diffs[indices]
    norms = np.linalg.norm(tang, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return tang / norms


def _build_frame_matrices(positions, pitch_deg, yaw_deg, roll_deg,
                          base_rotation=None):
    """Build (N, 4, 4) transforms combining translation + rotation, suitable
    for assigning directly to actor.user_matrix per animation frame.

    If `base_rotation` (3, 3) is given, it is applied on the left of every
    frame's rotation — used to mount the model onto the trajectory's initial
    heading so the data's per-frame angles read as attitude relative to that
    launch direction.
    """
    n = len(positions)
    rot = _euler_to_matrix(pitch_deg, yaw_deg, roll_deg)
    if base_rotation is not None:
        rot = base_rotation @ rot
    mats = np.tile(np.eye(4), (n, 1, 1))
    mats[:, :3, :3] = rot
    mats[:, :3, 3] = positions
    return mats


def flypath3d(data, line_width=1, color=None, colormap=None, 
              show_grid=True, show_axes=True, title=None,
              background='white', off_screen=False, return_plotter=False,
              animate=False, save_animation=None,
              model=None, pitch_col=3, yaw_col=4, roll_col=5,
              radians=False, order='pyr', speed=3.0, model_scale=1.0,
              xlim=None, ylim=None, zlim=None, z_scale=1.0):
    """MATLAB-like 3D trajectory plot with precision axis scaling.
    
    Renders a 3D trajectory with proper equal aspect ratio, MATLAB-style
    boxed axes with grid, and real-scale coordinate preservation.
    
    Parameters
    ----------
    data : str or (N,3) array-like
        Path to CSV file, or (N, 3) array of [x, y, z] points.
    line_width : int, default 2
        Width of the trajectory line.
    color : str, optional
        Single color for the trajectory line (e.g., 'blue', '#ff0000').
        Defaults to 'blue' if not specified.
    colormap : str, optional
        Matplotlib colormap name. Overrides color if provided.
    show_grid : bool, default True
        Show MATLAB-style boxed grid on all axes.
    show_axes : bool, default True
        Show axis labels and tick marks.
    title : str, optional
        Plot title.
    background : str, default 'white'
        Background color of the plotting window.
    off_screen : bool, default False
        If True, render off-screen (for saving to file without display).
    return_plotter : bool, default False
        If True, return the PyVista Plotter object instead of showing it.
    animate : bool, default False
        If True, animate a sphere moving along the trajectory path.
    save_animation : str, optional
        Path to save animation as a GIF.
    model : str or Path, optional
        Path to .mat file containing 3D model mesh (V, F, C arrays).
        If provided, uses model instead of sphere for animation.
    pitch_col : int, default 3
        Column index for pitch angle in trajectory data.
    yaw_col : int, default 4
        Column index for yaw angle in trajectory data.
    roll_col : int, default 5
        Column index for roll angle in trajectory data.
    radians : bool, default False
        If True, input angles are in radians and will be converted to degrees.
        If False (default), angles are assumed to be in degrees already.
    order : str, default 'pyr'
        Column order for orientation data: 'pyr' = pitch(3), yaw(4), roll(5).
        Other options: 'ypr', 'rpy', etc.
    speed : float, default 3.0
        Animation duration in seconds.
    
    Returns
    -------
    pyvista.Plotter or None
        If return_plotter is True, returns the Plotter object.
    """
    # Load data
    points, full_data, info = load_trajectory(data)
    
    if points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got shape {points.shape}")
    
    if len(points) < 2:
        raise ValueError(f"Need at least 2 points, got {len(points)}")
    
    # Determine orientation column indices
    # If auto-detected column map has values, use those instead of defaults
    col_map = info.get('col_map', {})
    if col_map.get('pitch') is not None:
        pitch_col = col_map['pitch']
    if col_map.get('yaw') is not None:
        yaw_col = col_map['yaw']
    if col_map.get('roll') is not None:
        roll_col = col_map['roll']
    
    # Determine color: default is 'blue', colormap overrides color
    use_color = color if color is not None else 'blue'
    use_colormap = colormap if colormap is not None else None
    
    # Create plotter
    # Offscreen only for GIF saving (screenshot-based). Interactive animation uses the window.
    is_save = save_animation is not None
    plotter = pv.Plotter(off_screen=off_screen or is_save)
    plotter.background_color = background
    
    # Create trajectory line as a spline for smoothness
    n_interp = max(len(points), len(points) * 10)
    spline = pv.Spline(points, n_interp)
    
    # Compute tube radius relative to data range (scales with data)
    data_range = np.ptp(points, axis=0).max()
    tube_radius = max(data_range * 0.0015 * line_width, 0.005)
    
    # Create tube mesh from the spline
    tube = spline.tube(radius=tube_radius)
    
    if use_colormap is not None:
        # Color by position along the path using a colormap
        n_pts = len(spline.points)
        position_scalar = np.linspace(0, 1, n_pts)
        spline.point_data['pos'] = position_scalar
        # Re-tube with scalars
        tube = spline.tube(radius=tube_radius)
        plotter.add_mesh(tube, scalars='pos', cmap=use_colormap,
                         smooth_shading=True, show_scalar_bar=False)
    else:
        plotter.add_mesh(tube, color=use_color, smooth_shading=True)
    
    # Marker radius: 0.5% of bounding box diagonal for consistent visual size
    bbox_diagonal = np.linalg.norm([data_range, data_range, data_range])
    marker_radius = bbox_diagonal * 0.005
    
    # Start marker: green sphere
    start_mesh = pv.Sphere(radius=marker_radius, center=points[0])
    plotter.add_mesh(start_mesh, color='green', smooth_shading=True)
    
    # End marker: red sphere
    end_mesh = pv.Sphere(radius=marker_radius, center=points[-1])
    plotter.add_mesh(end_mesh, color='red', smooth_shading=True)
    
    # Compute tight bounds with 10% padding (or use manual limits if provided)
    data_min = points.min(axis=0)
    data_max = points.max(axis=0)
    data_range_axes = data_max - data_min
    
    if xlim is not None:
        x_min, x_max = xlim
    else:
        x_padding = data_range_axes[0] * 0.1
        x_padding = max(x_padding, data_range * 0.01)
        x_min = data_min[0] - x_padding
        x_max = data_max[0] + x_padding
    
    if ylim is not None:
        y_min, y_max = ylim
    else:
        y_padding = data_range_axes[1] * 0.1
        y_padding = max(y_padding, data_range * 0.01)
        y_min = data_min[1] - y_padding
        y_max = data_max[1] + y_padding
    
    if zlim is not None:
        z_min, z_max = zlim
    else:
        z_padding = data_range_axes[2] * 0.1
        z_padding = max(z_padding, data_range * 0.01)
        z_min = data_min[2] - z_padding
        z_max = data_max[2] + z_padding
    
    bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
    
    # MATLAB-style boxed axes with grid and tight bounds
    if show_grid:
        plotter.show_grid(
            show_xaxis=show_axes, show_yaxis=show_axes, show_zaxis=show_axes,
            grid=True, location='outer', bold=True, font_size=10,
            bounds=bounds,
        )
    
    # Set title
    if title:
        plotter.add_text(title, position='upper_edge', font_size=14,
                         color='black' if background == 'white' else 'white')
    
    # Set camera to focus on the (possibly user-limited) bounds, not raw data,
    # so xlim/ylim/zlim actually reframe the view.
    bounds_center = np.array([(x_min + x_max) / 2,
                              (y_min + y_max) / 2,
                              (z_min + z_max) / 2])
    bounds_range_axes = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    max_range = max(bounds_range_axes) if max(bounds_range_axes) > 0 else 1.0

    # Set camera position
    plotter.camera_position = 'iso'
    plotter.camera.focal_point = bounds_center
    plotter.camera.position = (
        bounds_center[0] + max_range * 1.5,
        bounds_center[1] + max_range * 1.5,
        bounds_center[2] + max_range * 1.5,
    )
    plotter.camera.view_up = (0, 0, 1)

    # Set clipping range to data bounds so axes scale correctly
    plotter.camera.clipping_range = (0.01, max_range * 10)

    # Add axes labels
    if show_axes:
        plotter.add_axes()

    # Vertical exaggeration: visually stretch z while keeping real tick labels
    if z_scale != 1.0:
        plotter.set_scale(1, 1, z_scale, reset_camera=True)
    
    # --- Animation support ---
    if animate or save_animation is not None:
        sp = spline.points
        n_spline = len(sp)
        
        # Use ~30 fps for smooth animation
        n_frames = max(10, min(90, int(speed * 30)))
        frame_indices = np.linspace(0, n_spline - 1, n_frames, dtype=int)
        
        # Load 3D model if provided, otherwise use sphere
        if model is not None:
            # Load the 3D model mesh
            model_mesh = load_3d_model(model)
            
            # Scale model to appropriate size relative to trajectory
            model_bounds = model_mesh.bounds
            model_size = max(
                model_bounds[1] - model_bounds[0],  # x range
                model_bounds[3] - model_bounds[2],  # y range
                model_bounds[5] - model_bounds[4]   # z range
            )
            target_size = data_range * 0.05 * model_scale  # Model will be 5% of data range
            scale_factor = target_size / model_size if model_size > 0 else 1.0
            
            # Center the model at origin (create new mesh, avoid inplace)
            model_center = np.array(model_mesh.center)
            model_mesh = model_mesh.translate(-model_center)
            model_mesh = model_mesh.scale([scale_factor, scale_factor, scale_factor])
            # These meshes point nose along -X; flip 180 about the vertical axis
            # so the nose is +X, matching the alignment convention.
            model_mesh = model_mesh.rotate_z(180)

            # Add model to plotter
            anim_actor = plotter.add_mesh(
                model_mesh, color='gray',
            )
        else:
            # Animation sphere (same size as start/end markers)
            sphere_mesh = pv.Sphere(radius=marker_radius)
            anim_actor = plotter.add_mesh(
                sphere_mesh, color='yellow', smooth_shading=True,
                specular=0.5, specular_power=20
            )
        
        # Pre-compute all frame positions. Animated actors are positioned via
        # a user_matrix translation, which VTK does NOT scale with set_scale
        # (unlike baked-vertex meshes such as the tube/markers). So when the
        # scene is vertically exaggerated, pre-scale the z of moving actors to
        # keep them riding on the stretched trajectory.
        frame_positions = sp[frame_indices].copy()
        if z_scale != 1.0:
            frame_positions[:, 2] = frame_positions[:, 2] * z_scale

        # Get orientation data if available
        n_data = len(full_data) if full_data is not None else 0
        has_orientation = (full_data is not None and
                          n_data > max(pitch_col, yaw_col, roll_col))

        if has_orientation:
            # Compute frame indices aligned to original data length
            data_frame_indices = np.linspace(0, n_data - 1, n_frames, dtype=int)

            # Extract angles from the specified columns
            pitch_angles = full_data[data_frame_indices, pitch_col]
            yaw_angles = full_data[data_frame_indices, yaw_col]
            roll_angles = full_data[data_frame_indices, roll_col]
            # Convert radians to degrees if needed
            if radians or info.get('from_mat', False):
                pitch_angles = np.degrees(pitch_angles)
                yaw_angles = np.degrees(yaw_angles)
                roll_angles = np.degrees(roll_angles)

            # Mount the model onto the trajectory's initial heading, then apply
            # the data's per-frame attitude (which is expressed relative to that
            # launch direction) on top. This keeps the model aligned with the
            # trajectory at the start and through the turns.
            base_rotation = _alignment_rotation(points[min(4, len(points) - 1)]
                                                - points[0])
            frame_matrices = _build_frame_matrices(
                frame_positions, pitch_angles, yaw_angles, roll_angles,
                base_rotation=base_rotation)
        else:
            # No orientation data: point the model along the direction of travel
            # at every frame so it follows the path.
            tangents = _trajectory_tangents(sp, frame_indices)
            rots = np.stack([_alignment_rotation(t) for t in tangents])
            frame_matrices = np.tile(np.eye(4), (n_frames, 1, 1))
            frame_matrices[:, :3, :3] = rots
            frame_matrices[:, :3, 3] = frame_positions

        # Set initial position and orientation
        anim_actor.user_matrix = frame_matrices[0]
        
        if save_animation is not None:
            # For GIF saving: render each frame via screenshot (offscreen)
            import imageio
            frames = []
            for f_idx in range(n_frames):
                anim_actor.user_matrix = frame_matrices[f_idx]
                img = plotter.screenshot(return_img=True)
                frames.append(img)
            imageio.mimsave(save_animation, frames, fps=30, loop=0)
            print(f"Animation saved to {save_animation}")
        else:
            # Interactive animation: use VTK timer
            frame_index = [0]

            def update_frame(step):
                f_idx = frame_index[0] % n_frames
                anim_actor.user_matrix = frame_matrices[f_idx]
                frame_index[0] += 1
            
            duration_ms = max(10, int((speed * 1000) // n_frames))
            plotter.add_timer_event(100000, duration_ms, update_frame)
        
        if return_plotter:
            return plotter
        if not off_screen and save_animation is None:
            plotter.show()
        return None
    
    if return_plotter:
        return plotter
    
    if not off_screen:
        plotter.show()
    return None


def flypath3d_multi(trajectories, models=None, show_grid=True, show_axes=True,
                    title=None, background='white', off_screen=False,
                    return_plotter=False, animate=False, save_animation=None,
                    radians=False, order='pyr', speed=3.0, model_scale=1.0,
                    xlim=None, ylim=None, zlim=None, z_scale=1.0):
    """Plot multiple trajectories in the same 3D scene, optionally with 3D models.
    
    Parameters
    ----------
    trajectories : list of dict
        Each dict must have a 'data' key (path or array), and can optionally have:
        - 'color': str (e.g., 'red', 'blue', '#ff0000')
        - 'colormap': str (overrides color)
        - 'line_width': int (default 2)
        - 'label': str (legend label)
        - 'pitch_col', 'yaw_col', 'roll_col': int (orientation columns)
        - 'radians': bool (per-trajectory override)
    models : list of dict, optional
        Each dict specifies a 3D model to animate:
        - 'path': str (path to .mat model file)
        - 'trajectory_index': int (which trajectory this model follows)
        - 'color': str (model color, default 'gray')
    show_grid : bool, default True
    show_axes : bool, default True
    title : str, optional
    background : str, default 'white'
    off_screen : bool, default False
    return_plotter : bool, default False
    animate : bool, default False
    save_animation : str, optional
    radians : bool, default False
        If True, input angles are in radians. Default is degrees.
    order : str, default 'pyr'
        Column order for orientation data.
    speed : float, default 3.0
        Animation duration in seconds.
    
    Returns
    -------
    pyvista.Plotter or None
    """
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    
    # Compute global bounds for consistent scaling
    all_points = []
    all_infos = []
    for traj in trajectories:
        pts, _, info = load_trajectory(traj['data'])
        all_points.append(pts)
        all_infos.append(info)
    
    global_min = np.min([np.min(p, axis=0) for p in all_points], axis=0)
    global_max = np.max([np.max(p, axis=0) for p in all_points], axis=0)
    global_range = np.max(global_max - global_min)

    # Compute bounds with 10% padding (or use manual limits if provided)
    global_range_axes = global_max - global_min
    if xlim is not None:
        gx_min, gx_max = xlim
    else:
        pad = max(global_range_axes[0] * 0.1, global_range * 0.01)
        gx_min, gx_max = global_min[0] - pad, global_max[0] + pad
    if ylim is not None:
        gy_min, gy_max = ylim
    else:
        pad = max(global_range_axes[1] * 0.1, global_range * 0.01)
        gy_min, gy_max = global_min[1] - pad, global_max[1] + pad
    if zlim is not None:
        gz_min, gz_max = zlim
    else:
        pad = max(global_range_axes[2] * 0.1, global_range * 0.01)
        gz_min, gz_max = global_min[2] - pad, global_max[2] + pad
    global_bounds = [gx_min, gx_max, gy_min, gy_max, gz_min, gz_max]

    # Create plotter
    is_save = save_animation is not None
    plotter = pv.Plotter(off_screen=off_screen or is_save)
    plotter.background_color = background
    
    # Store spline data for animation
    spline_data = []
    
    for idx, (traj, points, info) in enumerate(zip(trajectories, all_points, all_infos)):
        color = traj.get('color', None)
        colormap = traj.get('colormap', None)
        line_width = traj.get('line_width', 2)
        label = traj.get('label', None)
        
        # Get column indices from auto-detection or user override
        col_map = info.get('col_map', {})
        pitch_col = traj.get('pitch_col', col_map.get('pitch', 3))
        yaw_col = traj.get('yaw_col', col_map.get('yaw', 4))
        roll_col = traj.get('roll_col', col_map.get('roll', 5))
        
        # Load full data for orientation
        data = traj['data']
        if isinstance(data, (str, Path)):
            from .io import load_csv, load_mat
            path = Path(data)
            if path.suffix.lower() == '.mat':
                full_data = load_mat(path)
            else:
                full_data, _, _ = load_csv(path)
                full_data = full_data[0]  # points array
        else:
            full_data = points
        
        # Determine color
        use_color = color if color is not None else 'blue'
        use_colormap = colormap if colormap is not None else None
        
        # Create spline
        n_interp = max(len(points), len(points) * 10)
        spline = pv.Spline(points, n_interp)
        
        # Tube radius relative to global range
        tube_radius = max(global_range * 0.0015 * line_width, 0.005)
        tube = spline.tube(radius=tube_radius)
        
        if use_colormap is not None:
            n_pts = len(spline.points)
            position_scalar = np.linspace(0, 1, n_pts)
            spline.point_data['pos'] = position_scalar
            tube = spline.tube(radius=tube_radius)
            plotter.add_mesh(tube, scalars='pos', cmap=use_colormap,
                             smooth_shading=True, show_scalar_bar=False,
                             label=label)
        else:
            plotter.add_mesh(tube, color=use_color, smooth_shading=True,
                             label=label)
        
        # Start/end markers
        bbox_diagonal = np.linalg.norm([global_range, global_range, global_range])
        marker_radius = bbox_diagonal * 0.005
        
        start_mesh = pv.Sphere(radius=marker_radius, center=points[0])
        plotter.add_mesh(start_mesh, color='green', smooth_shading=True)
        
        end_mesh = pv.Sphere(radius=marker_radius, center=points[-1])
        plotter.add_mesh(end_mesh, color='red', smooth_shading=True)
        
        spline_data.append({
            'spline': spline,
            'full_data': full_data,
            'pitch_col': pitch_col,
            'yaw_col': yaw_col,
            'roll_col': roll_col,
            'from_mat': info.get('from_mat', False),
            'points': points,
        })
    
    # Grid
    if show_grid:
        plotter.show_grid(
            show_xaxis=show_axes, show_yaxis=show_axes, show_zaxis=show_axes,
            grid=True, location='outer', bold=True, font_size=10,
            bounds=global_bounds,
        )

    # Title
    if title:
        plotter.add_text(title, position='upper_edge', font_size=14,
                         color='black' if background == 'white' else 'white')

    # Frame the (possibly user-limited) bounds so xlim/ylim/zlim reframe the view
    bounds_center = np.array([(gx_min + gx_max) / 2,
                              (gy_min + gy_max) / 2,
                              (gz_min + gz_max) / 2])
    bounds_range_axes = np.array([gx_max - gx_min, gy_max - gy_min, gz_max - gz_min])
    max_range = max(bounds_range_axes) if max(bounds_range_axes) > 0 else 1.0
    plotter.camera_position = 'iso'
    plotter.camera.focal_point = bounds_center
    plotter.camera.position = (
        bounds_center[0] + max_range * 1.5,
        bounds_center[1] + max_range * 1.5,
        bounds_center[2] + max_range * 1.5,
    )
    plotter.camera.view_up = (0, 0, 1)
    plotter.camera.clipping_range = (0.01, max_range * 10)

    if show_axes:
        plotter.add_axes()

    # Vertical exaggeration: visually stretch z while keeping real tick labels
    if z_scale != 1.0:
        plotter.set_scale(1, 1, z_scale, reset_camera=True)
    
    # Add legend if any labels
    if any(t.get('label') for t in trajectories):
        plotter.add_legend()
    
    # --- Animation support ---
    if animate or save_animation is not None:
        # Use the first trajectory's spline for frame count
        sp = spline_data[0]['spline'].points
        n_spline = len(sp)
        n_frames = max(10, min(90, int(speed * 30)))
        frame_indices = np.linspace(0, n_spline - 1, n_frames, dtype=int)
        
        # Pre-compute all frame data for each actor
        frame_data = []
        for idx, (traj, sd) in enumerate(zip(trajectories, spline_data)):
            sp_traj = sd['spline'].points
            # Use each trajectory's own spline length for frame indices
            n_local = len(sp_traj)
            local_frames = min(n_frames, n_local)
            local_indices = np.linspace(0, n_local - 1, local_frames, dtype=int)
            frame_positions = sp_traj[local_indices].copy()
            # See flypath3d(): moving actors need their z pre-scaled to ride the
            # vertically-exaggerated scene (set_scale doesn't move translations).
            if z_scale != 1.0:
                frame_positions[:, 2] = frame_positions[:, 2] * z_scale

            # Base mount rotation aligning the model's nose (+X) with this
            # trajectory's initial heading.
            base_rotation = _alignment_rotation(
                sd['points'][min(4, len(sd['points']) - 1)] - sd['points'][0])

            # Check if a model is assigned to this trajectory
            assigned_model = None
            model_color = 'gray'
            if models:
                for m in models:
                    if m.get('trajectory_index') == idx:
                        assigned_model = m['path']
                        model_color = m.get('color', 'gray')
                        break
            
            if assigned_model is not None:
                # Load and scale model
                model_mesh = load_3d_model(assigned_model)
                model_bounds = model_mesh.bounds
                model_size = max(
                    model_bounds[1] - model_bounds[0],
                    model_bounds[3] - model_bounds[2],
                    model_bounds[5] - model_bounds[4]
                )
                target_size = global_range * 0.05
                scale_factor = target_size / model_size if model_size > 0 else 1.0
                
                model_center = np.array(model_mesh.center)
                model_mesh = model_mesh.translate(-model_center)
                model_mesh = model_mesh.scale([scale_factor, scale_factor, scale_factor])
                # Nose points -X in these meshes; flip so nose is +X.
                model_mesh = model_mesh.rotate_z(180)

                actor = plotter.add_mesh(
                    model_mesh, color=model_color,
                )
                
                # Pre-compute orientations
                fd = sd['full_data']
                n_data = len(fd) if fd is not None else 0
                has_orient = (fd is not None and 
                             n_data > max(sd['pitch_col'], sd['yaw_col'], sd['roll_col']))
                
                if has_orient:
                    data_frame_indices = np.linspace(0, n_data - 1, local_frames, dtype=int)
                    pitch_angles = fd[data_frame_indices, sd['pitch_col']]
                    yaw_angles = fd[data_frame_indices, sd['yaw_col']]
                    roll_angles = fd[data_frame_indices, sd['roll_col']]
                    
                    if radians or sd['from_mat']:
                        pitch_angles = np.degrees(pitch_angles)
                        yaw_angles = np.degrees(yaw_angles)
                        roll_angles = np.degrees(roll_angles)

                    # Mount onto the initial heading, then apply the data's
                    # per-frame attitude relative to it.
                    frame_matrices = _build_frame_matrices(
                        frame_positions, pitch_angles, yaw_angles, roll_angles,
                        base_rotation=base_rotation)
                else:
                    # No attitude data: point the model along the direction of
                    # travel at every frame.
                    tangents = _trajectory_tangents(sp_traj, local_indices)
                    rots = np.stack([_alignment_rotation(t) for t in tangents])
                    frame_matrices = np.tile(np.eye(4), (local_frames, 1, 1))
                    frame_matrices[:, :3, :3] = rots
                    frame_matrices[:, :3, 3] = frame_positions

                frame_data.append({
                    'actor': actor,
                    'positions': frame_positions,
                    'matrices': frame_matrices,
                    'is_model': True,
                })
            else:
                # Use sphere
                bbox_diagonal = np.linalg.norm([global_range, global_range, global_range])
                marker_radius = bbox_diagonal * 0.005
                sphere_mesh = pv.Sphere(radius=marker_radius)
                sphere_color = traj.get('color', 'yellow')
                actor = plotter.add_mesh(
                    sphere_mesh, color=sphere_color, smooth_shading=True,
                    specular=0.5, specular_power=20
                )
                frame_data.append({
                    'actor': actor,
                    'positions': frame_positions,
                    'is_model': False,
                })
        
        if save_animation is not None:
            # For GIF saving: render each frame via screenshot (offscreen)
            import imageio
            frames = []
            for f_idx in range(n_frames):
                for fd in frame_data:
                    n_local = len(fd['positions'])
                    li = min(f_idx, n_local - 1)
                    if fd['is_model']:
                        fd['actor'].user_matrix = fd['matrices'][li]
                    else:
                        pos = fd['positions'][li]
                        fd['actor'].SetPosition(pos[0], pos[1], pos[2])
                img = plotter.screenshot(return_img=True)
                frames.append(img)
            imageio.mimsave(save_animation, frames, fps=30, loop=0)
            print(f"Animation saved to {save_animation}")
        else:
            # Interactive animation: use VTK timer
            frame_index = [0]
            
            def update_frame(step):
                f_idx = frame_index[0] % n_frames
                for fd in frame_data:
                    n_local = len(fd['positions'])
                    li = min(f_idx, n_local - 1)
                    if fd['is_model']:
                        fd['actor'].user_matrix = fd['matrices'][li]
                    else:
                        pos = fd['positions'][li]
                        fd['actor'].SetPosition(pos[0], pos[1], pos[2])
                frame_index[0] += 1
            
            duration_ms = max(10, int((speed * 1000) // n_frames))
            plotter.add_timer_event(100000, duration_ms, update_frame)
        
        if return_plotter:
            return plotter
        if not off_screen and save_animation is None:
            plotter.show()
        return None
    
    if return_plotter:
        return plotter
    
    if not off_screen:
        plotter.show()
    return None


def plot_trajectory(*args, **kwargs):
    """Alias for flypath3d."""
    return flypath3d(*args, **kwargs)