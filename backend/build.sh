#!/bin/bash
set -e  # Exit on error

echo "Initializing git submodules..."
git submodule update --init --recursive

echo "Installing backend requirements..."
pip install -r requirements.txt

echo "Installing CommandGenerator package..."
pip install -e ../dataset_generator/CommandGenerator

echo "Build completed successfully!"
