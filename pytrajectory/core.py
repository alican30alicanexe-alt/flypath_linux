"""
Core plotting functions for pytrajectory.
MATLAB-like 3D trajectory visualization using PyVista.
"""

import numpy as np
import pyvista as pv
from pathlib import Path
from .io import load_trajectory, load_3d_model


def _make_spline(points, max_ctrl=1000, max_interp=6000):
    """Build a smooth spline through the path.

    pv.Spline fits a parametric spline whose cost explodes with the number of
    control points, so dense trajectories (tens of thousands of points) are
    downsampled to at most `max_ctrl` control points first — otherwise building
    the spline effectively hangs. The interpolated resolution is also capped.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n > max_ctrl:
        points = points[np.linspace(0, n - 1, max_ctrl, dtype=int)]
    n_interp = min(max(len(points), len(points) * 10), max_interp)
    return pv.Spline(points, n_interp)


def _prepare_model_mesh(path, target_size):
    """Load a model mesh, anchor it on its body axis, scale it to `target_size`,
    and flip it so its nose points +X (the meshes here point -X).

    The anchor (the point placed on the trajectory) is the model's TAIL along
    the longest axis — so the tail stays on the current path point (where the
    trail ends) regardless of scale, and the body extends forward with the nose
    ahead — but the vertex centroid across the other two axes, so the path runs
    through the body centerline rather than the bounding-box center. Otherwise a
    protrusion like an aircraft's tail fin lifts the box center above the
    fuselage, and the path (and trail) floats over the model.

    The meshes point nose along -X (tail = +X, the maximum bound); after the
    rotate_z(180) below the tail sits at the origin and the nose points +X.
    """
    mesh = load_3d_model(path)
    b = mesh.bounds
    extents = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    size = extents.max()
    factor = target_size / size if size > 0 else 1.0
    long_axis = int(np.argmax(extents))
    anchor = np.array(mesh.points).mean(axis=0)          # body centroid
    anchor[long_axis] = b[2 * long_axis + 1]             # tail (max bound)
    mesh = mesh.translate(-anchor)
    mesh = mesh.scale([factor, factor, factor])
    return mesh.rotate_z(180)


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


def _mount_base(init_dir, full_data, pitch_col, yaw_col, from_mat,
                yaw_sign=-1.0, pitch_sign=-1.0, radians=False):
    """Base rotation that mounts the model onto the trajectory's initial
    heading.

    Aligns the model nose (+X) with the initial direction of travel, then undoes
    the data's attitude at the first sample. This matters when the data's zero
    reference is not the initial tangent (e.g. trajectory_enemy starts with a
    nonzero yaw): without it a constant heading offset remains for the whole
    flight.
    """
    base_align = _alignment_rotation(init_dir)
    if full_data is None:
        return base_align
    n_data = len(full_data)
    if n_data <= max(pitch_col, yaw_col):
        return base_align
    p0 = full_data[0, pitch_col]
    y0 = full_data[0, yaw_col]
    if radians or from_mat:
        p0, y0 = np.degrees(p0), np.degrees(y0)
    r0 = _euler_to_matrix([pitch_sign * p0], [yaw_sign * y0], [0.0])[0]
    return base_align @ r0.T


def _face_is_path(face):
    """True if the model should face the path tangent (velocity) rather than the
    data's attitude. Useful for long/thin projectiles whose attitude data wobbles
    around the smooth path and slaloms when scaled up."""
    return str(face).lower() in ('path', 'velocity', 'tangent', 'vel')


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


def _model_matrices_along(sd, spline_indices, yaw_sign=-1.0, pitch_sign=-1.0,
                          roll_sign=1.0, radians=False, face='data'):
    """Compute (N, 4, 4) placement transforms for a model at the given spline
    indices of a trajectory, oriented by the trajectory's attitude data (or by
    the direction of travel when no attitude data is present, or when
    face='path'). Shared by the static multi-placement path."""
    sp = sd['spline'].points
    n_spline = len(sp)
    positions = sp[spline_indices]
    init_dir = sd['points'][min(4, len(sd['points']) - 1)] - sd['points'][0]

    fd = sd['full_data']
    n_data = len(fd) if fd is not None else 0
    has_orient = (fd is not None and
                  n_data > max(sd['pitch_col'], sd['yaw_col'], sd['roll_col'])
                  and not _face_is_path(face))

    if has_orient:
        base_rotation = _mount_base(init_dir, fd, sd['pitch_col'], sd['yaw_col'],
                                    sd['from_mat'], yaw_sign, pitch_sign, radians)
        fracs = np.asarray(spline_indices, dtype=float) / max(1, n_spline - 1)
        di = np.round(fracs * (n_data - 1)).astype(int)
        pitch = fd[di, sd['pitch_col']]
        yaw = fd[di, sd['yaw_col']]
        roll = fd[di, sd['roll_col']]
        if radians or sd['from_mat']:
            pitch, yaw, roll = np.degrees(pitch), np.degrees(yaw), np.degrees(roll)
        pitch, yaw, roll = pitch_sign * pitch, yaw_sign * yaw, roll_sign * roll
        return _build_frame_matrices(positions, pitch, yaw, roll,
                                     base_rotation=base_rotation)

    tangents = _trajectory_tangents(sp, spline_indices)
    rots = np.stack([_alignment_rotation(t) for t in tangents])
    mats = np.tile(np.eye(4), (len(positions), 1, 1))
    mats[:, :3, :3] = rots
    mats[:, :3, 3] = positions
    return mats


def flypath3d(data, line_width=50, color=None, colormap=None,
              show_grid=True, show_axes=True, title=None,
              background='white', off_screen=False, return_plotter=False,
              animate=False, save_animation=None,
              model=None, pitch_col=3, yaw_col=4, roll_col=5,
              radians=False, order='pyr', speed=3.0, model_scale=1.0,
              xlim=None, ylim=None, zlim=None, z_scale=1.0,
              yaw_sign=-1.0, pitch_sign=-1.0, roll_sign=1.0,
              show_markers=False, trail=False, face='data'):
    """MATLAB-like 3D trajectory plot with precision axis scaling.
    
    Renders a 3D trajectory with proper equal aspect ratio, MATLAB-style
    boxed axes with grid, and real-scale coordinate preservation.
    
    Parameters
    ----------
    data : str or (N,3) array-like
        Path to CSV file, or (N, 3) array of [x, y, z] points.
    line_width : int, default 50
        Width of the trajectory line on a fine scale (100 ~= the classic
        default thickness; the default 50 is a moderately thin line).
    show_markers : bool, default False
        Show green start / red end markers. Hidden by default.
    trail : bool, default False
        During animation, reveal the trajectory line progressively as the model
        passes over it (like a real flight trail) instead of drawing it upfront.
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
    
    # Create trajectory line as a spline for smoothness (downsampled control
    # points so dense trajectories don't stall the spline fit).
    spline = _make_spline(points)

    # Compute tube radius relative to data range (scales with data)
    data_range = np.ptp(points, axis=0).max()
    # Fine-grained: line_width=100 ~= the classic default thickness; the
    # default (15) is a thin line.
    tube_radius = max(data_range * 7e-6 * line_width, 1e-4)

    # When animating with a trail, the line is revealed progressively as the
    # model passes over it, so skip drawing the full static tube up front.
    animating = animate or save_animation is not None
    draw_full_line = not (trail and animating)

    if draw_full_line:
        tube = spline.tube(radius=tube_radius)
        if use_colormap is not None:
            # Color by position along the path using a colormap
            n_pts = len(spline.points)
            spline.point_data['pos'] = np.linspace(0, 1, n_pts)
            tube = spline.tube(radius=tube_radius)
            plotter.add_mesh(tube, scalars='pos', cmap=use_colormap,
                             smooth_shading=True, show_scalar_bar=False)
        else:
            plotter.add_mesh(tube, color=use_color, smooth_shading=True)

    # Marker radius: 0.5% of bounding box diagonal for consistent visual size
    bbox_diagonal = np.linalg.norm([data_range, data_range, data_range])
    marker_radius = bbox_diagonal * 0.005

    # Start/end markers (green/red) are hidden unless requested.
    if show_markers:
        plotter.add_mesh(pv.Sphere(radius=marker_radius, center=points[0]),
                         color='green', smooth_shading=True)
        plotter.add_mesh(pv.Sphere(radius=marker_radius, center=points[-1]),
                         color='red', smooth_shading=True)
    
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

        # Progressive trail: a tube that grows behind the model as it moves.
        # Use a downsampled path so rebuilding the tube each frame stays cheap.
        # The model is anchored at its tail, so the trail ends at the current
        # position (the tail); for a plain sphere marker pull it back a touch.
        trail_actor = None
        if trail:
            n_trail = min(n_spline, 300)
            trail_pts = sp[np.linspace(0, n_spline - 1, n_trail, dtype=int)]
            total_len = np.linalg.norm(np.diff(trail_pts, axis=0), axis=1).sum()
            back = 0.0 if model is not None else marker_radius
            frac_off = (back / total_len) if total_len > 0 else 0.0

            def _trail_tube(frac):
                frac = max(0.0, frac - frac_off)
                k = max(2, int(round(frac * (n_trail - 1))) + 1)
                return pv.Spline(trail_pts[:k]).tube(radius=tube_radius)
            trail_actor = plotter.add_mesh(_trail_tube(0.0), color=use_color,
                                           smooth_shading=True)

        # Load 3D model if provided, otherwise use sphere
        if model is not None:
            # Load, anchor on the body axis, and scale to ~5% of the data range.
            target_size = data_range * 0.05 * model_scale
            model_mesh = _prepare_model_mesh(model, target_size)

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
                          n_data > max(pitch_col, yaw_col, roll_col)
                          and not _face_is_path(face))

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

            # Map the input angle convention onto the render matrices. The
            # bundled flight data (and MATLAB flypath3d) use the aerospace
            # convention: yaw positive = nose right (compass/clockwise), pitch
            # positive = nose up. Our matrices are math-convention, so yaw and
            # pitch are negated by default. Flip a sign for data that differs.
            pitch_angles = pitch_sign * pitch_angles
            yaw_angles = yaw_sign * yaw_angles
            roll_angles = roll_sign * roll_angles

            # Mount the model onto the trajectory's initial heading (accounting
            # for the data's attitude at the first sample), then apply the
            # per-frame attitude on top, keeping it aligned through the turns.
            base_rotation = _mount_base(
                points[min(4, len(points) - 1)] - points[0], full_data,
                pitch_col, yaw_col, info.get('from_mat', False),
                yaw_sign, pitch_sign, radians)
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
                if trail_actor is not None:
                    trail_actor.mapper.dataset = _trail_tube(frame_indices[f_idx] / max(1, n_spline - 1))
                # screenshot() alone does not re-render after a transform change,
                # so force a render each frame or every frame would be identical.
                plotter.render()
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
                if trail_actor is not None:
                    trail_actor.mapper.dataset = _trail_tube(frame_indices[f_idx] / max(1, n_spline - 1))
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
                    xlim=None, ylim=None, zlim=None, z_scale=1.0,
                    yaw_sign=-1.0, pitch_sign=-1.0, roll_sign=1.0,
                    show_markers=False, trail=False,
                    view=None, window_size=None, face='data',
                    grid_kwargs=None, zoom=None):
    """Plot multiple trajectories in the same 3D scene, optionally with 3D models.

    view : str or (azimuth, elevation), optional
        Camera preset: 'top' (look straight down, x up / y right), 'side', or
        'iso' (default). A tuple sets an explicit azimuth/elevation.
    window_size : (w, h), optional
        Render window size in pixels (e.g. (1000, 280) for a wide scene).
    grid_kwargs : dict, optional
        Extra keyword arguments merged into the `show_grid()` call, e.g.
        {'xtitle': 'X (m)', 'n_xlabels': 7, 'fmt': '%.0f'}.
    zoom : float, optional
        Camera zoom applied after framing. Values below 1.0 pull back, leaving
        room for the tick labels that sit outside the bounding box.

    models : list of dict, optional
        Each dict may also include:
        - 'scale': float (per-model size multiplier, overrides model_scale)
        - 'count': int (place this many static copies along the path when not
          animating — a multi-exposure view of the trajectory)
    
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
    if window_size is not None:
        plotter.window_size = list(window_size)

    # Store spline data for animation
    spline_data = []
    any_label = False  # track whether a labeled mesh was actually drawn

    for idx, (traj, points, info) in enumerate(zip(trajectories, all_points, all_infos)):
        color = traj.get('color', None)
        colormap = traj.get('colormap', None)
        line_width = traj.get('line_width', 50)
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
        
        # Create spline (downsampled control points to keep the fit fast)
        spline = _make_spline(points)

        # Tube radius relative to global range
        tube_radius = max(global_range * 7e-6 * line_width, 1e-4)

        # Skip the full static tube when it will be revealed as a trail.
        animating = animate or save_animation is not None
        if not (trail and animating):
            tube = spline.tube(radius=tube_radius)
            if use_colormap is not None:
                n_pts = len(spline.points)
                spline.point_data['pos'] = np.linspace(0, 1, n_pts)
                tube = spline.tube(radius=tube_radius)
                plotter.add_mesh(tube, scalars='pos', cmap=use_colormap,
                                 smooth_shading=True, show_scalar_bar=False,
                                 label=label)
            else:
                plotter.add_mesh(tube, color=use_color, smooth_shading=True,
                                 label=label)
            if label:
                any_label = True

        # Start/end markers (hidden unless requested)
        bbox_diagonal = np.linalg.norm([global_range, global_range, global_range])
        marker_radius = bbox_diagonal * 0.005
        if show_markers:
            plotter.add_mesh(pv.Sphere(radius=marker_radius, center=points[0]),
                             color='green', smooth_shading=True)
            plotter.add_mesh(pv.Sphere(radius=marker_radius, center=points[-1]),
                             color='red', smooth_shading=True)
        
        spline_data.append({
            'spline': spline,
            'full_data': full_data,
            'pitch_col': pitch_col,
            'yaw_col': yaw_col,
            'roll_col': roll_col,
            'from_mat': info.get('from_mat', False),
            'points': points,
            'tube_radius': tube_radius,
            'color': use_color,
        })
    
    # Grid
    def _draw_grid():
        plotter.show_grid(
            show_xaxis=show_axes, show_yaxis=show_axes, show_zaxis=show_axes,
            grid=True, location='outer', bold=True, font_size=10,
            bounds=global_bounds,
            **(grid_kwargs or {}),
        )

    if show_grid:
        _draw_grid()

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
    if view in ('top', 'xy'):
        # Look straight down the -Z axis: x points up, y points right.
        # Parallel projection gives a flat, MATLAB-like orthographic view.
        plotter.enable_parallel_projection()
        plotter.camera.focal_point = bounds_center
        plotter.camera.position = (bounds_center[0], bounds_center[1],
                                   bounds_center[2] + max_range * 2.0)
        plotter.camera.view_up = (1, 0, 0)
    elif view in ('side', 'xz'):
        plotter.enable_parallel_projection()
        plotter.camera.focal_point = bounds_center
        plotter.camera.position = (bounds_center[0], bounds_center[1] - max_range * 2.0,
                                   bounds_center[2])
        plotter.camera.view_up = (0, 0, 1)
    else:
        plotter.camera_position = 'iso'
        plotter.camera.focal_point = bounds_center
        plotter.camera.position = (
            bounds_center[0] + max_range * 1.5,
            bounds_center[1] + max_range * 1.5,
            bounds_center[2] + max_range * 1.5,
        )
        plotter.camera.view_up = (0, 0, 1)
    plotter.camera.clipping_range = (0.01, max_range * 10)

    if zoom is not None:
        plotter.camera.zoom(zoom)

    if show_axes:
        plotter.add_axes()

    # Vertical exaggeration: visually stretch z while keeping real tick labels
    if z_scale != 1.0:
        plotter.set_scale(1, 1, z_scale, reset_camera=True)
    
    # Add legend if any labeled meshes were drawn (trail mode draws none upfront)
    if any_label:
        plotter.add_legend()

    # --- Static multi-placement (multi-exposure): place N model copies along
    # each assigned path when not animating (mirrors flypath3d's `step`). ---
    animating = animate or save_animation is not None
    if models and not animating:
        for m in models:
            idx = m.get('trajectory_index', 0)
            if idx >= len(spline_data):
                continue
            sd = spline_data[idx]
            count = max(1, int(m.get('count', 1)))
            scale = m.get('scale', model_scale)
            color = m.get('color', 'gray')
            target_size = global_range * 0.05 * scale
            base_mesh = _prepare_model_mesh(m['path'], target_size)

            n_sp = len(sd['spline'].points)
            place_idx = np.linspace(0, n_sp - 1, count, dtype=int)
            traj_face = trajectories[idx].get('face', face)
            mats = _model_matrices_along(sd, place_idx, yaw_sign, pitch_sign,
                                         roll_sign, radians, traj_face)
            if z_scale != 1.0:
                mats[:, 2, 3] = mats[:, 2, 3] * z_scale
            for k in range(count):
                actor = plotter.add_mesh(base_mesh.copy(), color=color)
                actor.user_matrix = mats[k]

    # --- Animation support ---
    if animate or save_animation is not None:
        # Use the first trajectory's spline for frame count
        sp = spline_data[0]['spline'].points
        n_spline = len(sp)
        n_frames = max(10, min(90, int(speed * 30)))
        frame_indices = np.linspace(0, n_spline - 1, n_frames, dtype=int)
        
        # Pre-compute all frame data for each actor
        frame_data = []
        trails = []  # (actor, sp_traj, local_indices, tube_radius) for progressive reveal
        for idx, (traj, sd) in enumerate(zip(trajectories, spline_data)):
            sp_traj = sd['spline'].points
            # Use each trajectory's own spline length for frame indices
            n_local = len(sp_traj)
            local_frames = min(n_frames, n_local)
            local_indices = np.linspace(0, n_local - 1, local_frames, dtype=int)
            frame_positions = sp_traj[local_indices].copy()

            # Model assigned to this trajectory (needed for the trail tail offset)
            assigned_scale = None
            for m in (models or []):
                if m.get('trajectory_index') == idx:
                    assigned_scale = m.get('scale', model_scale)
                    break

            # Progressive trail actor for this trajectory (downsampled path so
            # rebuilding the tube each frame stays cheap). The model is anchored
            # at its tail, so the trail ends at the current position (the tail).
            if trail:
                tr = sd['tube_radius']
                n_t = min(len(sp_traj), 300)
                tpts = sp_traj[np.linspace(0, len(sp_traj) - 1, n_t, dtype=int)]
                total_len = np.linalg.norm(np.diff(tpts, axis=0), axis=1).sum()
                back = 0.0 if assigned_scale is not None else global_range * 0.005
                frac_off = (back / total_len) if total_len > 0 else 0.0
                tube0 = pv.Spline(tpts[:2]).tube(radius=tr)
                trail_actor = plotter.add_mesh(tube0, color=sd['color'],
                                               smooth_shading=True)
                trails.append((trail_actor, tpts, n_t, tr, frac_off))
            # See flypath3d(): moving actors need their z pre-scaled to ride the
            # vertically-exaggerated scene (set_scale doesn't move translations).
            if z_scale != 1.0:
                frame_positions[:, 2] = frame_positions[:, 2] * z_scale

            # Check if a model is assigned to this trajectory
            assigned_model = None
            model_color = 'gray'
            model_scale_local = model_scale
            if models:
                for m in models:
                    if m.get('trajectory_index') == idx:
                        assigned_model = m['path']
                        model_color = m.get('color', 'gray')
                        model_scale_local = m.get('scale', model_scale)
                        break

            if assigned_model is not None:
                # Load, scale and orient the model (nose -> +X)
                target_size = global_range * 0.05 * model_scale_local
                model_mesh = _prepare_model_mesh(assigned_model, target_size)

                actor = plotter.add_mesh(
                    model_mesh, color=model_color,
                )
                
                # Pre-compute orientations
                fd = sd['full_data']
                n_data = len(fd) if fd is not None else 0
                traj_face = traj.get('face', face)
                has_orient = (fd is not None and
                             n_data > max(sd['pitch_col'], sd['yaw_col'], sd['roll_col'])
                             and not _face_is_path(traj_face))

                if has_orient:
                    # Mount rotation aligning the model's nose (+X) with the
                    # initial heading (accounting for its start attitude).
                    base_rotation = _mount_base(
                        sd['points'][min(4, len(sd['points']) - 1)] - sd['points'][0],
                        sd['full_data'], sd['pitch_col'], sd['yaw_col'],
                        sd['from_mat'], yaw_sign, pitch_sign, radians)
                    data_frame_indices = np.linspace(0, n_data - 1, local_frames, dtype=int)
                    pitch_angles = fd[data_frame_indices, sd['pitch_col']]
                    yaw_angles = fd[data_frame_indices, sd['yaw_col']]
                    roll_angles = fd[data_frame_indices, sd['roll_col']]
                    
                    if radians or sd['from_mat']:
                        pitch_angles = np.degrees(pitch_angles)
                        yaw_angles = np.degrees(yaw_angles)
                        roll_angles = np.degrees(roll_angles)

                    # Apply the input-convention signs (see flypath3d()).
                    pitch_angles = pitch_sign * pitch_angles
                    yaw_angles = yaw_sign * yaw_angles
                    roll_angles = roll_sign * roll_angles

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

                # Seat the model on its first frame straight away. The prepared
                # mesh sits at the origin until a user_matrix moves it, so an
                # actor left unplaced stretches the renderer's bounds all the
                # way back to (0,0,0) — which the outer axes then fit to, giving
                # an animation axes far larger than the data.
                actor.user_matrix = frame_matrices[0]

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

        # Model meshes are authored at the origin and only moved by user_matrix,
        # so add_mesh() registers each one at (0,0,0) and the grid — created
        # above, before any of these actors existed — unions that in, drawing
        # axes that run all the way back to zero. Every actor is now seated on
        # its first frame, so redrawing the grid here picks up the true extent.
        if show_grid:
            plotter.remove_bounds_axes()
            _draw_grid()

        def _grow_trails(f_idx):
            frac = f_idx / max(1, n_frames - 1)
            for actor, tpts, n_t, tr, frac_off in trails:
                fr = max(0.0, frac - frac_off)
                k = max(2, int(round(fr * (n_t - 1))) + 1)
                actor.mapper.dataset = pv.Spline(tpts[:k]).tube(radius=tr)

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
                _grow_trails(f_idx)
                # Force a render; screenshot() alone won't reflect the updates.
                plotter.render()
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
                _grow_trails(f_idx)
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