"""
Demo script for pytrajectory.
Shows various ways to use the flypath3d function.
"""

import numpy as np
from pytrajectory import flypath3d, load_csv


def demo_from_csv():
    """Example 1: Load trajectory from CSV file."""
    print("Running demo: trajectory from CSV...")
    flypath3d('examples/sample_data.csv', title='Trajectory from CSV')


def demo_spiral():
    """Example 2: 3D spiral trajectory from numpy array."""
    print("Running demo: 3D spiral...")
    t = np.linspace(0, 6 * np.pi, 200)
    x = t * np.cos(t) / 5
    y = t * np.sin(t) / 5
    z = t / 10
    points = np.column_stack([x, y, z])
    flypath3d(points, title='3D Spiral Trajectory', colormap='viridis')


def demo_helix():
    """Example 3: Helix trajectory."""
    print("Running demo: helix...")
    t = np.linspace(0, 4 * np.pi, 150)
    points = np.column_stack([
        np.cos(t),
        np.sin(t),
        t / (2 * np.pi)
    ])
    flypath3d(points, title='Helix Trajectory', color='blue')


def demo_figure8():
    """Example 4: Figure-8 trajectory."""
    print("Running demo: figure-8...")
    t = np.linspace(0, 2 * np.pi, 200)
    points = np.column_stack([
        np.sin(t),
        np.sin(2 * t) / 2,
        np.cos(t) / 3
    ])
    flypath3d(points, title='Figure-8 Trajectory', colormap='plasma')


if __name__ == '__main__':
    print("pytrajectory Demo")
    print("=" * 40)
    print("Select a demo:")
    print("  1 - CSV file trajectory")
    print("  2 - 3D Spiral")
    print("  3 - Helix")
    print("  4 - Figure-8")
    print("  all - Run all demos (one at a time)")
    
    choice = input("Choice (1-4, or 'all'): ").strip()
    
    demos = {
        '1': demo_from_csv,
        '2': demo_spiral,
        '3': demo_helix,
        '4': demo_figure8,
        'all': lambda: [demo_from_csv(), demo_spiral(), demo_helix(), demo_figure8()],
    }
    
    if choice in demos:
        demos[choice]()
    else:
        print("Invalid choice. Running spiral demo by default.")
        demo_spiral()