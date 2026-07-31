#!/bin/bash
# Build script for creating a standalone pytrajectory executable.
#
# Usage:
#     chmod +x build.sh
#     ./build.sh
#
# Requirements:
#     pip install pyinstaller pyinstaller-hooks-contrib
#
# The resulting binary will be in: dist/pytrajectory/pytrajectory

set -e  # Exit on error

echo "=== pytrajectory Build Script ==="
echo ""

# Activate virtual environment if one exists (checked in order of preference)
for venv_dir in venv .venv trajectory.venv; do
    if [ -f "$venv_dir/bin/activate" ]; then
        echo "Activating virtual environment: $venv_dir"
        source "$venv_dir/bin/activate"
        break
    fi
done

# Check if pyinstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "Error: pyinstaller not found."
    echo "Install it with: pip install pyinstaller pyinstaller-hooks-contrib"
    exit 1
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
echo "To run with a bundled model (from the dist directory):"
echo "  cd dist/pytrajectory"
echo "  ./pytrajectory ../../examples/sample_data.csv --model ../../models/f-16.mat --scale 10 --animate"
