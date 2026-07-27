"""
Command-line interface for pytrajectory.
Provides a CLI executable for plotting 3D trajectories.
"""
import sys
import argparse
import numpy as np
from .core import flypath3d, flypath3d_multi
from .io import load_csv


def demo_spiral(animate=False, save_animation=None):
    """3D spiral trajectory."""
    t = np.linspace(0, 6 * np.pi, 200)
    x = t * np.cos(t) / 5
    y = t * np.sin(t) / 5
    z = t / 10
    points = np.column_stack([x, y, z])
    flypath3d(points, title='3D Spiral Trajectory', colormap='viridis',
              animate=animate, save_animation=save_animation)


def demo_helix(animate=False, save_animation=None):
    """Helix trajectory."""
    t = np.linspace(0, 4 * np.pi, 150)
    points = np.column_stack([
        np.cos(t),
        np.sin(t),
        t / (2 * np.pi)
    ])
    flypath3d(points, title='Helix Trajectory', color='blue',
              animate=animate, save_animation=save_animation)


def demo_figure8(animate=False, save_animation=None):
    """Figure-8 trajectory."""
    t = np.linspace(0, 2 * np.pi, 200)
    points = np.column_stack([
        np.sin(t),
        np.sin(2 * t) / 2,
        np.cos(t) / 3
    ])
    flypath3d(points, title='Figure-8 Trajectory', colormap='plasma',
              animate=animate, save_animation=save_animation)


def demo_multi(animate=False, save_animation=None):
    """Multi-trajectory demo: spiral + helix together with models."""
    from .core import flypath3d_multi
    
    # Create two trajectories
    t1 = np.linspace(0, 6 * np.pi, 200)
    spiral = np.column_stack([
        t1 * np.cos(t1) / 5,
        t1 * np.sin(t1) / 5,
        t1 / 10
    ])
    
    t2 = np.linspace(0, 4 * np.pi, 150)
    helix = np.column_stack([
        np.cos(t2) + 2,
        np.sin(t2) + 2,
        t2 / (2 * np.pi)
    ])
    
    trajectories = [
        {'data': spiral, 'color': 'red', 'label': 'Spiral', 'line_width': 3},
        {'data': helix, 'color': 'blue', 'label': 'Helix'},
    ]
    
    flypath3d_multi(
        trajectories,
        title='Multi-Trajectory Demo',
        animate=animate,
        save_animation=save_animation,
    )


def _parse_color(value):
    """Return a color usable by PyVista: an (r, g, b) tuple for 'r,g,b' strings,
    otherwise the string unchanged (name or hex)."""
    if value is None:
        return None
    if ',' in value:
        try:
            return tuple(float(c) for c in value.split(','))
        except ValueError:
            return value
    return value


def parse_traj_blocks(blocks, defaults):
    """Parse repeatable --traj blocks into (trajectories, models).

    Each block is a token list: the first token is the trajectory file path and
    the rest are key=value options. Recognized keys: color, colormap, label,
    model, mcolor, scale, lw/linewidth/line-width. Missing options fall back to
    the `defaults` dict (from the global --color/--colormap/--model/--scale/
    --line-width flags).

    Returns dicts in the shapes flypath3d_multi consumes.
    """
    trajectories = []
    models = []
    for idx, block in enumerate(blocks):
        if not block:
            continue
        path = block[0]
        opts = {}
        for tok in block[1:]:
            if '=' not in tok:
                raise ValueError(
                    f"--traj option '{tok}' must be key=value (block: {path})")
            key, val = tok.split('=', 1)
            opts[key.strip().lower()] = val

        color = _parse_color(opts.get('color', defaults.get('color')))
        colormap = opts.get('colormap', defaults.get('colormap'))
        label = opts.get('label')
        lw = opts.get('lw', opts.get('linewidth', opts.get('line-width')))
        line_width = int(lw) if lw is not None else defaults.get('line_width')

        traj = {'data': path}
        if colormap is not None:
            traj['colormap'] = colormap
        elif color is not None:
            traj['color'] = color
        if line_width is not None:
            traj['line_width'] = line_width
        if label:
            traj['label'] = label
        trajectories.append(traj)

        model_path = opts.get('model', defaults.get('model'))
        if model_path:
            scale = opts.get('scale', defaults.get('scale'))
            models.append({
                'path': model_path,
                'trajectory_index': idx,
                'color': _parse_color(opts.get('mcolor', 'gray')),
                'scale': float(scale) if scale is not None else 1.0,
            })

    return trajectories, (models or None)


def main():
    parser = argparse.ArgumentParser(
        description='pytrajectory - 3D trajectory visualization (MATLAB-like)',
        epilog='Examples:\n'
               '  pytrajectory trajectory.csv\n'
               '  pytrajectory trajectory.csv --color blue\n'
               '  pytrajectory trajectory.csv --model models/f-16.mat --scale 10 --animate\n'
               '  pytrajectory --demo spiral\n'
               '  pytrajectory --list-demos\n'
               '  # Multiple trajectories: one --traj block each (path then key=value opts)\n'
               '  pytrajectory \\\n'
               '    --traj enemy.mat  color=red   model=models/f-16.mat scale=10 lw=20 \\\n'
               '    --traj friend.mat color=blue  model=models/f-16.mat scale=10 \\\n'
               '    --traj aam.mat    color=black model=models/missile.mat mcolor=yellow \\\n'
               '    --animate --trail\n'
               '  # keys: color, colormap, label, model, mcolor, scale, lw\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single trajectory options
    parser.add_argument(
        'file', nargs='?', default=None,
        help='Path to CSV file with x, y, z columns'
    )
    parser.add_argument(
        '--color', default=None,
        help='Line color (e.g., "blue", "#ff0000")'
    )
    parser.add_argument(
        '--colormap', default=None,
        help='Colormap name (e.g., "jet", "viridis", "plasma")'
    )
    parser.add_argument(
        '--line-width', type=int, default=15,
        help='Trajectory line width on a fine scale (default: 15, a thin line; '
             '100 ~= the classic thickness)'
    )
    parser.add_argument(
        '--markers', action='store_true',
        help='Show start (green) and end (red) markers (hidden by default)'
    )
    parser.add_argument(
        '--trail', action='store_true',
        help='Reveal the trajectory line progressively as the model flies over '
             'it (like a real flight trail); requires --animate'
    )
    parser.add_argument(
        '--title', default=None,
        help='Plot title'
    )
    parser.add_argument(
        '--no-grid', action='store_true',
        help='Hide grid'
    )
    parser.add_argument(
        '--no-axes', action='store_true',
        help='Hide axes labels'
    )
    parser.add_argument(
        '--background', default='white',
        help='Background color (default: white)'
    )
    parser.add_argument(
        '--demo', nargs='?', const='spiral', default=None,
        choices=['spiral', 'helix', 'figure8', 'multi', 'all'],
        help='Run a demo trajectory (default: spiral)'
    )
    parser.add_argument(
        '--list-demos', action='store_true',
        help='List available demo names'
    )
    parser.add_argument(
        '--animate', action='store_true',
        help='Animate sphere moving along the trajectory (completes in ~3 sec)'
    )
    parser.add_argument(
        '--model', default=None,
        help='Path to .mat file with 3D model (V, F, C arrays)'
    )
    parser.add_argument(
        '--pitch-col', type=int, default=3,
        help='Column index for pitch angle in trajectory data (default: 3)'
    )
    parser.add_argument(
        '--yaw-col', type=int, default=4,
        help='Column index for yaw angle in trajectory data (default: 4)'
    )
    parser.add_argument(
        '--roll-col', type=int, default=5,
        help='Column index for roll angle in trajectory data (default: 5)'
    )
    parser.add_argument(
        '--rad', action='store_true',
        help='Input angles are in radians (convert to degrees)'
    )
    parser.add_argument(
        '--order', default='pyr',
        help='Column order for orientation: "pyr"=pitch(3)/yaw(4)/roll(5) (default)'
    )
    parser.add_argument(
        '--speed', type=float, default=3.0,
        help='Animation duration in seconds (default: 3.0)'
    )
    parser.add_argument(
        '--model-scale', type=float, default=1.0,
        help='Scale factor for 3D model size (default: 1.0)'
    )
    parser.add_argument(
        '--scale', type=float, default=None,
        help='Model size (alias for --model-scale; also the default scale for '
             '--traj blocks). Wins over --model-scale if both are given.'
    )
    parser.add_argument(
        '--xlim', type=float, nargs=2, default=None,
        help='X-axis limits: min max (e.g., --xlim 0 1000)'
    )
    parser.add_argument(
        '--ylim', type=float, nargs=2, default=None,
        help='Y-axis limits: min max (e.g., --ylim 10000 15000)'
    )
    parser.add_argument(
        '--zlim', type=float, nargs=2, default=None,
        help='Z-axis limits: min max (e.g., --zlim 0 500)'
    )
    parser.add_argument(
        '--math', action='store_true',
        help='Use the math angle convention (yaw CCW-positive, pitch nose-down-'
             'positive) instead of the default aerospace convention (yaw '
             'compass/clockwise-positive, pitch nose-up-positive)'
    )
    parser.add_argument(
        '--flip-yaw', action='store_true',
        help='Invert yaw sign (use if the model turns the wrong way / too far)'
    )
    parser.add_argument(
        '--flip-pitch', action='store_true',
        help='Invert pitch sign (use if the model pitches the wrong way)'
    )
    parser.add_argument(
        '--flip-roll', action='store_true',
        help='Invert roll sign (use if the model banks the wrong way)'
    )
    parser.add_argument(
        '--z-scale', type=float, default=1.0,
        help='Vertical exaggeration factor for the z-axis (default: 1.0, no change). '
             'E.g. --z-scale 5 stretches z 5x for flat/altitude-limited trajectories '
             'while keeping real z tick labels.'
    )
    parser.add_argument(
        '--save-animation', default=None,
        help='Save animation as a GIF file'
    )
    parser.add_argument(
        '--screenshot', default=None,
        help='Save screenshot to file (requires --off-screen)'
    )
    parser.add_argument(
        '--off-screen', action='store_true',
        help='Render off-screen (no window)'
    )

    # Multi-trajectory: one repeatable --traj block per trajectory.
    parser.add_argument(
        '--traj', action='append', nargs='+', metavar='PATH KEY=VALUE',
        help='Add a trajectory: a file path followed by key=value options '
             '(color, colormap, label, model, mcolor, scale, lw). Repeat for '
             'multiple trajectories. Global --color/--colormap/--model/--scale/'
             '--line-width act as defaults.'
    )

    args = parser.parse_args()

    # Resolve angle-convention signs. Default is aerospace; --math switches to
    # the math convention; --flip-* invert individual axes on top.
    yaw_sign = 1.0 if args.math else -1.0
    pitch_sign = 1.0 if args.math else -1.0
    roll_sign = 1.0
    if args.flip_yaw:
        yaw_sign = -yaw_sign
    if args.flip_pitch:
        pitch_sign = -pitch_sign
    if args.flip_roll:
        roll_sign = -roll_sign

    # Model size: --scale is an alias for --model-scale and wins if given.
    model_scale = args.scale if args.scale is not None else args.model_scale

    # Handle --list-demos
    if args.list_demos:
        print("Available demos:")
        print("  spiral  - 3D spiral trajectory")
        print("  helix   - Helix trajectory")
        print("  figure8 - Figure-8 trajectory")
        print("  multi   - Multiple trajectories together")
        print("  all     - Run all demos sequentially")
        return

    # Handle --demo
    if args.demo:
        demos = {
            'spiral': lambda: demo_spiral(animate=args.animate, save_animation=args.save_animation),
            'helix': lambda: demo_helix(animate=args.animate, save_animation=args.save_animation),
            'figure8': lambda: demo_figure8(animate=args.animate, save_animation=args.save_animation),
            'multi': lambda: demo_multi(animate=args.animate, save_animation=args.save_animation),
            'all': lambda: [demo_spiral(animate=args.animate, save_animation=args.save_animation),
                            demo_helix(animate=args.animate, save_animation=args.save_animation),
                            demo_figure8(animate=args.animate, save_animation=args.save_animation),
                            demo_multi(animate=args.animate, save_animation=args.save_animation)],
        }
        print(f"Running demo: {args.demo}")
        demos[args.demo]()
        return

    # Handle --traj (one or more trajectories)
    if args.traj:
        defaults = {
            'color': args.color,
            'colormap': args.colormap,
            'model': args.model,
            'scale': args.scale,
            'line_width': args.line_width,
        }
        trajectories, models = parse_traj_blocks(args.traj, defaults)
        flypath3d_multi(
            trajectories,
            models=models,
            show_grid=not args.no_grid,
            show_axes=not args.no_axes,
            title=args.title,
            background=args.background,
            off_screen=args.off_screen,
            animate=args.animate,
            save_animation=args.save_animation,
            speed=args.speed,
            model_scale=model_scale,
            xlim=args.xlim,
            ylim=args.ylim,
            zlim=args.zlim,
            z_scale=args.z_scale,
            yaw_sign=yaw_sign,
            pitch_sign=pitch_sign,
            roll_sign=roll_sign,
            show_markers=args.markers,
            trail=args.trail,
        )
        return

    # Handle single CSV file input
    if args.file:
        flypath3d(
            args.file,
            line_width=args.line_width,
            color=args.color,
            colormap=args.colormap,
            show_grid=not args.no_grid,
            show_axes=not args.no_axes,
            title=args.title,
            background=args.background,
            off_screen=args.off_screen,
            animate=args.animate,
            save_animation=args.save_animation,
            model=args.model,
            pitch_col=args.pitch_col,
            yaw_col=args.yaw_col,
            roll_col=args.roll_col,
            radians=args.rad,
            order=args.order,
            speed=args.speed,
            model_scale=model_scale,
            xlim=args.xlim,
            ylim=args.ylim,
            zlim=args.zlim,
            z_scale=args.z_scale,
            yaw_sign=yaw_sign,
            pitch_sign=pitch_sign,
            roll_sign=roll_sign,
            show_markers=args.markers,
            trail=args.trail,
        )
        return

    # No arguments — show help
    parser.print_help()


if __name__ == '__main__':
    main()