#!/bin/bash

# Script to run the inference container with automatic GPU detection

echo "Detecting hardware capabilities..."

# Detect operating system
OS=$(uname -s)

# Check if NVIDIA GPU is available and Docker can use nvidia runtime
if command -v nvidia-smi >/dev/null 2>&1 && docker info | grep -q "nvidia"; then
    echo "✓ NVIDIA GPU detected and Docker nvidia runtime available"
    echo "Starting container with GPU acceleration..."
    
    # Create temporary override file for GPU support
    if [ "$OS" = "Linux" ]; then
        echo "✓ Linux detected - using host networking for optimal performance"
        cat > docker-compose.gpu.yml << EOF
services:
  ollama:
    runtime: nvidia
    network_mode: host
EOF
    else
        echo "✓ Non-Linux OS detected - using port mapping"
        cat > docker-compose.gpu.yml << EOF
services:
  ollama:
    runtime: nvidia
EOF
    fi
    
    # Run with GPU support
    docker compose -f ../docker-compose.yml -f docker-compose.gpu.yml up "$@"
    
    # Clean up temporary file
    rm -f docker-compose.gpu.yml
    
else
    # Handle CPU-only mode (no GPU or no container toolkit)
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "ℹ NVIDIA GPU detected but Docker nvidia runtime not available"
        echo "  See README for NVIDIA Container Toolkit installation instructions"
    else
        echo "ℹ No NVIDIA GPU detected"
    fi
    echo "Starting container with CPU mode..."
    
    # Create OS-specific override for CPU mode
    if [ "$OS" = "Linux" ]; then
        echo "✓ Linux detected - using host networking for optimal performance"
        cat > docker-compose.override.yml << EOF
services:
  ollama:
    network_mode: host
EOF
        docker compose -f ../docker-compose.yml -f docker-compose.override.yml up "$@"
        rm -f docker-compose.override.yml
    elif [ "$OS" = "Darwin" ]; then
        echo "✓ macOS detected - optimizing for Apple Silicon"
        cat > docker-compose.override.yml << EOF
services:
  ollama:
    platform: linux/arm64
EOF
        docker compose -f ../docker-compose.yml -f docker-compose.override.yml up "$@"
        rm -f docker-compose.override.yml
    else
        docker compose -f ../docker-compose.yml up "$@"
    fi
fi 