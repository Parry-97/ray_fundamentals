#!/bin/bash

# Set Ray memory environment variables
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_memory_usage_threshold=0.6
export RAY_DISABLE_IMPORT_WARNING=1

# Check available memory before running
echo "=== System Memory Check ==="
free -h

echo ""
echo "=== Running Ray script with memory limits ==="

# Run the Python script with uv
uv run ray_advanced.py