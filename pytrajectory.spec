# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for building pytrajectory as a standalone executable.

Usage:
    pyinstaller pytrajectory.spec

The resulting binary will be in dist/pytrajectory/
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Collect all model files
model_files = []
models_dir = os.path.join(project_root, 'models')
if os.path.isdir(models_dir):
    for f in os.listdir(models_dir):
        if f.endswith('.mat'):
            model_files.append((os.path.join(models_dir, f), 'models'))

# Collect example CSV files
example_files = []
examples_dir = os.path.join(project_root, 'examples')
if os.path.isdir(examples_dir):
    for f in os.listdir(examples_dir):
        if f.endswith('.csv') or f.endswith('.obj'):
            example_files.append((os.path.join(examples_dir, f), 'examples'))

# Combine all data files
datas = model_files + example_files

a = Analysis(
    ['pytrajectory/__main__.py'],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'pytrajectory',
        'pytrajectory.core',
        'pytrajectory.cli',
        'pytrajectory.io',
        'numpy',
        'pyvista',
        'pandas',
        'scipy.io',
        'scipy.spatial',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'PIL',
        'test',
        'distutils',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pytrajectory',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)