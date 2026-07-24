"""
Core plotting functions for pytrajectory.
MATLAB-like 3D trajectory visualization using PyVista.
"""

import numpy as np
import pyvista as pv
from pathlib import Path
from .io import load_trajectory, load_3d_model


def _get_orientation_columns(order_str):
    """Parse order string into (pitch_col, yaw_col, roll_col) column indices.
    
    Default aerospace standard: 'pyr' = pitch(3), yaw(4), roll(5)
    Other options: 'ypr' = yaw(3), pitch(4), roll(5), etc.
    """
    order_map = {
        'p': 3,  # pitch
        'y': 4,  # yaw  
        'r': 5,  # roll
    }
    result = {}
    for i, ch in enumerate(order_str.lower()[:3]):
        if ch in order_map:
            result[ch] = order_map[ch]
    return result.get('p', 3), result.get('y', 4), result.get('r', 5)


def flypath3d(data, line_width=2, color=None, colormap=None, 
              show_grid=True, show_axes=True, title=None,
              background='white', off_screen=False, return_plotter=False,
              animate=False, save_animation=None,
              model=None, pitch_col=3, yaw_col=4, roll_col=5,
              radians=False, order='pyr'):
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
        If True, animate a sphere moving along the trajectory path
        (completes in ~3 seconds).
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
    
    # MATLAB-style boxed axes with grid
    if show_grid:
        plotter.show_grid(
            show_xaxis=show_axes, show_yaxis=show_axes, show_zaxis=show_axes,
            grid=True, location='outer', bold=True, font_size=10,
        )
    
    # Set title
    if title:
        plotter.add_text(title, position='upper_edge', font_size=14,
                         color='black' if background == 'white' else 'white')
    
    # Set isometric view
    plotter.view_isometric()
    
    # Add axes labels
    if show_axes:
        plotter.add_axes()
    
    # --- Animation support ---
    if animate or save_animation is not None:
        sp = spline.points
        n_spline = len(sp)
        
        # Use fixed ~90 frames for consistent ~3 second animation
        n_frames = min(90, n_spline)
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
            target_size = data_range * 0.05  # Model will be 5% of data range
            scale_factor = target_size / model_size if model_size > 0 else 1.0
            
            # Center the model at origin (create new mesh, avoid inplace)
            model_center = np.array(model_mesh.center)
            model_mesh = model_mesh.translate(-model_center)
            model_mesh = model_mesh.scale([scale_factor, scale_factor, scale_factor])
            
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
        
        # Pre-compute all frame positions and orientations
        frame_positions = sp[frame_indices]
        
        # Get orientation data if available
        has_orientation = (full_data is not None and 
                          full_data.shape[1] > max(pitch_col, yaw_col, roll_col))
        
        if has_orientation:
            # Extract angles from the specified columns
            pitch_angles = full_data[frame_indices, pitch_col]
            yaw_angles = full_data[frame_indices, yaw_col]
            roll_angles = full_data[frame_indices, roll_col]
            
            # Convert radians to degrees if needed
            if radians or info.get('from_mat', False):
                # .mat files from MATLAB store angles in radians
                pitch_angles = np.degrees(pitch_angles)
                yaw_angles = np.degrees(yaw_angles)
                roll_angles = np.degrees(roll_angles)
            
            # Apply aerospace rotation order: VTK SetOrientation(roll, pitch, yaw)
            # applies ZYX intrinsic (yaw -> pitch -> roll)
            frame_orientations = np.column_stack([roll_angles, pitch_angles, yaw_angles])
        else:
            frame_orientations = np.zeros((n_frames, 3))
        
        # Set initial position
        anim_actor.SetPosition(frame_positions[0][0], frame_positions[0][1], frame_positions[0][2])
        if has_orientation:
            anim_actor.SetOrientation(*frame_orientations[0])
        
        if save_animation is not None:
            # For GIF saving: render each frame via screenshot (offscreen)
            import imageio
            frames = []
            for f_idx in range(n_frames):
                pos = frame_positions[f_idx]
                if has_orientation:
                    anim_actor.SetOrientation(*frame_orientations[f_idx])
                anim_actor.SetPosition(pos[0], pos[1], pos[2])
                img = plotter.screenshot(return_img=True)
                frames.append(img)
            imageio.mimsave(save_animation, frames, fps=30, loop=0)
            print(f"Animation saved to {save_animation}")
        else:
            # Interactive animation: use VTK timer
            frame_index = [0]
            
            def update_frame(step):
                f_idx = frame_index[0] % n_frames
                pos = frame_positions[f_idx]
                if has_orientation:
                    anim_actor.SetOrientation(*frame_orientations[f_idx])
                anim_actor.SetPosition(pos[0], pos[1], pos[2])
                frame_index[0] += 1
            
            duration_ms = max(10, 3000 // n_frames)
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
                    radians=False, order='pyr'):
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
        })
    
    # Grid
    if show_grid:
        plotter.show_grid(
            show_xaxis=show_axes, show_yaxis=show_axes, show_zaxis=show_axes,
            grid=True, location='outer', bold=True, font_size=10,
        )
    
    # Title
    if title:
        plotter.add_text(title, position='upper_edge', font_size=14,
                         color='black' if background == 'white' else 'white')
    
    plotter.view_isometric()
    
    if show_axes:
        plotter.add_axes()
    
    # Add legend if any labels
    if any(t.get('label') for t in trajectories):
        plotter.add_legend()
    
    # --- Animation support ---
    if animate or save_animation is not None:
        # Use the first trajectory's spline for frame count
        sp = spline_data[0]['spline'].points
        n_spline = len(sp)
        n_frames = min(90, n_spline)
        frame_indices = np.linspace(0, n_spline - 1, n_frames, dtype=int)
        
        # Pre-compute all frame data for each actor
        frame_data = []
        for idx, (traj, sd) in enumerate(zip(trajectories, spline_data)):
            sp_traj = sd['spline'].points
            # Use each trajectory's own spline length for frame indices
            n_local = len(sp_traj)
            local_frames = min(n_frames, n_local)
            local_indices = np.linspace(0, n_local - 1, local_frames, dtype=int)
            frame_positions = sp_traj[local_indices]
            
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
                
                actor = plotter.add_mesh(
                    model_mesh, color=model_color,
                )
                
                # Pre-compute orientations
                fd = sd['full_data']
                has_orient = (fd is not None and 
                             fd.shape[1] > max(sd['pitch_col'], sd['yaw_col'], sd['roll_col']))
                
                if has_orient:
                    pitch_angles = fd[frame_indices, sd['pitch_col']]
                    yaw_angles = fd[frame_indices, sd['yaw_col']]
                    roll_angles = fd[frame_indices, sd['roll_col']]
                    
                    if radians or sd['from_mat']:
                        pitch_angles = np.degrees(pitch_angles)
                        yaw_angles = np.degrees(yaw_angles)
                        roll_angles = np.degrees(roll_angles)
                    
                    frame_orientations = np.column_stack([roll_angles, pitch_angles, yaw_angles])
                else:
                    frame_orientations = np.zeros((n_frames, 3))
                
                frame_data.append({
                    'actor': actor,
                    'positions': frame_positions,
                    'orientations': frame_orientations,
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
                    pos = fd['positions'][f_idx]
                    if fd['is_model']:
                        fd['actor'].SetOrientation(*fd['orientations'][f_idx])
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
                    pos = fd['positions'][f_idx]
                    if fd['is_model']:
                        fd['actor'].SetOrientation(*fd['orientations'][f_idx])
                    fd['actor'].SetPosition(pos[0], pos[1], pos[2])
                frame_index[0] += 1
            
            duration_ms = max(10, 3000 // n_frames)
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