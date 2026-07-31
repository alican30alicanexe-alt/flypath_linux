"""Entry point for the standalone PyInstaller build. Kept outside the
pytrajectory package so it can use an absolute import (a script run directly
by the bootloader has no package context for relative imports).
"""
from pytrajectory.cli import main

if __name__ == '__main__':
    main()
