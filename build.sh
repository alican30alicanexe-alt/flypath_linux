#!/bin/bash
"""
Build script for creating a standalone pytrajectory executable.

Usage:
    chmod +x build.sh
    ./build.sh

Requirements:
    pip install pyinstaller

The resulting binary will be in: dist/pytrajectory
"""

set -e  # Exit on error

echo "=== pytrajectory Build Script ==="
echo ""

# Check if pyinstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "Error: pyinstaller not found."
    echo "Install it with: pip install pyinstaller"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "trajectory.venv" ]; then
    echo "Activating virtual environment..."
    source trajectory.venv/bin/activate
fi

echo "Step 1: Installing pytrajectory in development mode..."
pip install -e .

echo ""
echo "Step 2: Building standalone executable with PyInstaller..."
pyinstaller pytrajectory.spec --clean

echo ""
echo "=== Build Complete! ==="
echo "Executable: dist/pytrajectory/pytrajectory"
echo ""
echo "Test it with:"
echo "  ./dist/pytrajectory/pytrajectory --demo spiral"
echo "  ./dist/pytrajectory/pytrajectory --list-demos"
echo ""
echo "To run with models (from the dist directory):"
echo "  cd dist/pytrajectory"
echo "  ./pytrajectory --multi ../examples/sample_data.csv:red --models ../models/f-16.mat:0:gray"