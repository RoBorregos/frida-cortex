#!/bin/bash
set -e  # Exit on error

echo "Initializing git submodules..."
git submodule update --init --recursive

echo "Installing backend requirements..."
# Use minimal requirements if DISABLE_EMBEDDINGS is set
if [ "${DISABLE_EMBEDDINGS}" = "true" ]; then
    echo "Using minimal requirements (embeddings disabled)..."
    pip install -r requirements-minimal.txt
else
    echo "Using full requirements (embeddings enabled)..."
    pip install -r requirements.txt
fi

echo "Installing CommandGenerator package..."
pip install -e ../dataset_generator/CommandGenerator

echo "Build completed successfully!"
