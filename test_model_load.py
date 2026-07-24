#!/usr/bin/env python3
from pytrajectory.io import load_3d_model

print("Testing F-16 model loading...")
mesh = load_3d_model('models/f-16.mat')
print(f"F-16 Model loaded: {mesh.n_points} vertices, {mesh.n_faces} faces")

print("\nTesting missile model loading...")
mesh2 = load_3d_model('models/missile.mat')
print(f"Missile Model loaded: {mesh2.n_points} vertices, {mesh2.n_faces} faces")

print("\nAll models loaded successfully!")