#!/usr/bin/env python3
"""Diagnostic script to find what causes the double-free crash."""
import sys
import numpy as np

print("=" * 60)
print("DIAGNOSTIC: pytrajectory crash test")
print("=" * 60)

# Test 1: Load model
print("\n[1/5] Testing model loading...")
try:
    from pytrajectory.io import load_3d_model
    mesh = load_3d_model('models/f-16.mat')
    print(f"  OK: Model loaded: {mesh.n_points} vertices, {mesh.n_faces} faces")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 2: Load CSV
print("\n[2/5] Testing CSV loading...")
try:
    from pytrajectory.io import load_csv
    points, df = load_csv('examples/sample_data.csv')
    print(f"  OK: CSV loaded: {len(points)} points")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 3: Create plotter and add trajectory (no animation)
print("\n[3/5] Testing static plot (no animation, no model)...")
try:
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True)
    spline = pv.Spline(points, len(points) * 10)
    tube = spline.tube(radius=0.01)
    plotter.add_mesh(tube, color='blue')
    plotter.show_grid()
    plotter.view_isometric()
    # Just create it, don't show
    plotter.close()
    print("  OK: Static plot created and closed")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 4: Create plotter with model mesh (no animation)
print("\n[4/5] Testing plot with model mesh (no animation)...")
try:
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True)
    spline = pv.Spline(points, len(points) * 10)
    tube = spline.tube(radius=0.01)
    plotter.add_mesh(tube, color='blue')
    
    # Add model
    model_mesh = load_3d_model('models/f-16.mat')
    # Scale it
    data_range = np.ptp(points, axis=0).max()
    target_size = data_range * 0.05
    model_bounds = model_mesh.bounds
    model_size = max(
        model_bounds[1] - model_bounds[0],
        model_bounds[3] - model_bounds[2],
        model_bounds[5] - model_bounds[4]
    )
    scale_factor = target_size / model_size if model_size > 0 else 1.0
    model_center = np.array(model_mesh.center)
    model_mesh.translate(-model_center, inplace=True)
    model_mesh.scale([scale_factor, scale_factor, scale_factor], inplace=True)
    
    actor = plotter.add_mesh(model_mesh, color='gray')
    plotter.show_grid()
    plotter.view_isometric()
    plotter.close()
    print("  OK: Plot with model created and closed")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 5: Simple animation without model
print("\n[5/5] Testing simple animation (no model)...")
try:
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True)
    spline = pv.Spline(points, len(points) * 10)
    tube = spline.tube(radius=0.01)
    plotter.add_mesh(tube, color='blue')
    
    sp = spline.points
    n_frames = min(30, len(sp))
    frame_indices = np.linspace(0, len(sp) - 1, n_frames, dtype=int)
    
    # Add sphere
    sphere = pv.Sphere(radius=0.05)
    sphere_actor = plotter.add_mesh(sphere, color='yellow')
    sphere_actor.SetPosition(sp[0][0], sp[0][1], sp[0][2])
    
    plotter.show_grid()
    plotter.view_isometric()
    plotter.show(auto_close=False)
    
    for f_idx in range(n_frames):
        pos = sp[frame_indices[f_idx]]
        sphere_actor.SetPosition(pos[0], pos[1], pos[2])
        plotter.render()
    
    plotter.close()
    print("  OK: Simple animation completed")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED - No crash detected")
print("=" * 60)