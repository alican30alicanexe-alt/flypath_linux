# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for building a standalone pytrajectory executable that runs
without a Python installation. Bundles the CLI, its bundled 3D models and
example data, and the PyVista/VTK rendering stack.

Build with:
    pyinstaller pytrajectory.spec --clean
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None

hiddenimports = (
    collect_submodules('vtkmodules')
    + collect_submodules('pyvista')
    + collect_submodules('scipy.io')
    + [
        'imageio',
        'imageio.plugins.pillow',
    ]
)

datas = (
    [
        ('models', 'models'),
        ('examples', 'examples'),
    ]
    + collect_data_files('pyvista')
    + collect_data_files('vtkmodules')
    + copy_metadata('imageio')
)

a = Analysis(
    ['pytrajectory_launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pytrajectory',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='pytrajectory',
)
