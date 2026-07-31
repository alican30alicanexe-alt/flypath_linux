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

# models/ and examples/ are copied into dist/pytrajectory/ separately (see
# build.sh / the Windows CI workflow) rather than bundled here. PyInstaller
# 6+ hides datas inside a "_internal" subfolder by default; disabling that
# (contents_directory='.') instead collides with our own top-level
# `pytrajectory` package once the executable — also named `pytrajectory` —
# sits flat in the same directory, breaking `import pytrajectory.core`. A
# plain post-build copy sidesteps the whole _internal/collision question.
datas = (
    collect_data_files('pyvista')
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
