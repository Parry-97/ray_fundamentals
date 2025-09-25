#!/bin/bash

echo "=== Ray Cleanup Script ==="

# Show current Ray temp usage
echo "Current Ray temp directory usage:"
du -sh /tmp/ray* 2>/dev/null || echo "No Ray temp directories found"

# Stop any running Ray processes
echo ""
echo "Stopping any running Ray processes..."
ray stop 2>/dev/null || echo "No Ray processes to stop"

# Clean up Ray temporary directories
echo ""
echo "Cleaning up Ray temporary directories..."
rm -rf /tmp/ray*
rm -rf /tmp/ray_custom*

# Show disk space after cleanup
echo ""
echo "Disk space after cleanup:"
df -h /tmp

echo ""
echo "Ray cleanup completed!"