#!/usr/bin/env python3
"""
Minimal Ray test to verify Ray works with memory constraints.
"""

import numpy as np
import ray

# Initialize Ray with very conservative memory limits
ray.init(
    object_store_memory=128_000_000,   # 128MB for object store
    _memory=256_000_000,               # 256MB total memory limit
    num_cpus=1,                        # Use only 1 CPU core
    ignore_reinit_error=True
)

print("Ray initialized successfully!")
print("Available resources:", ray.available_resources())

# Test 1: Simple remote function
@ray.remote
def simple_task(x):
    return x * 2

# Test with small data
result = ray.get(simple_task.remote(42))
print(f"Simple task result: {result}")

# Test 2: Small matrix operations
small_matrix = np.random.rand(10, 10)  # Very small matrix
matrix_ref = ray.put(small_matrix)
retrieved_matrix = ray.get(matrix_ref)

print(f"Small matrix test passed: {np.array_equal(small_matrix, retrieved_matrix)}")

# Cleanup
ray.shutdown()
print("Ray shutdown successfully!")