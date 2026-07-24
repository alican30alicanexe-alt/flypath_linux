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


def parse_multi_arg(multi_args):
    """Parse --multi arguments in format: file:color:label or file:color or just file.
    
    Returns list of trajectory dicts.
    """
    trajectories = []
    for arg in multi_args:
        parts = arg.split(':')
        filepath = parts[0]
        color = parts[1] if len(parts) > 1 else None
        label = parts[2] if len(parts) > 2 else None
        traj = {'data': filepath, 'color': color}
        if label:
            traj['label'] = label
        trajectories.append(traj)
    return trajectories


def parse_model_arg(model_args):
    """Parse --model arguments in format: file:traj_idx:color.
    
    Returns list of model dicts.
    """
    models = []
    for arg in model_args:
        parts = arg.split(':')
        path = parts[0]
        traj_idx = int(parts[1]) if len(parts) > 1 else 0
        color = parts[2] if len(parts) > 2 else 'gray'
        models.append({
            'path': path,
            'trajectory_index': traj_idx,
            'color': color,
        })
    return models


def main():
    parser = argparse.ArgumentParser(
        description='pytrajectory - 3D trajectory visualization (MATLAB-like)',
        epilog='Examples:\n'
               '  pytrajectory trajectory.csv\n'
               '  pytrajectory trajectory.csv --color blue\n'
               '  pytrajectory --demo spiral\n'
               '  pytrajectory --demo all\n'
               '  pytrajectory --list-demos\n'
               '  pytrajectory --multi traj1.csv:red traj2.csv:blue\n'
               '  pytrajectory --multi missle.csv:red:Missile --models f16.mat:0:gray\n',
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
        '--line-width', type=int, default=2,
        help='Trajectory line width (default: 2)'
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

    # Multi-trajectory options
    parser.add_argument(
        '--multi', nargs='+', default=None,
        help='Multiple trajectories: file:color:label file:color ...'
    )
    parser.add_argument(
        '--models', nargs='+', default=None,
        help='3D models for multi mode: file:traj_idx:color file:idx:color ...'
    )

    args = parser.parse_args()

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

    # Handle --multi (multiple trajectories)
    if args.multi:
        trajectories = parse_multi_arg(args.multi)
        models = parse_model_arg(args.models) if args.models else None
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
        )
        return

    # No arguments — show help
    parser.print_help()


if __name__ == '__main__':
    main()