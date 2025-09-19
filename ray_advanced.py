import sys
import time
import os
import shutil

import numpy as np
import ray

# Create a custom temporary directory for Ray
RAY_TEMP_DIR = "/tmp/ray_custom"
if os.path.exists(RAY_TEMP_DIR):
    shutil.rmtree(RAY_TEMP_DIR)  # Clean up previous runs
os.makedirs(RAY_TEMP_DIR, exist_ok=True)

# Initialize Ray with memory limits and custom temp directory
ray.init(
    object_store_memory=512_000_000,  # 512MB for object store
    _memory=1_000_000_000,  # 1GB total memory limit
    num_cpus=1,  # Use only 1 CPU core
    _temp_dir=RAY_TEMP_DIR,  # Use custom temp directory (note the underscore)
    ignore_reinit_error=True,  # Allow re-initialization
)

# INFO: Object Store
# Each worker node has its own object store, and collectively, these form a shared
# object store across the cluster.
# Remote objects are immutable. That is, their values cannot be changed after creation.
# This allows remote objects to be replicated in multiple object stores without needing to synchronize the copies.


# INFO: Tasks and actors create and work with remote objects,
# which can be stored anywhere in a cluster. These objects are
# accessed using ObjectRef and are cached in a distributed
# shared-memory object store.Let's consider the following example:

large_matrix = np.random.rand(
    256, 256, 32
)  # approx. 64 MB (further reduced for memory constraints)
size_in_bytes = sys.getsizeof(large_matrix)

# NOTE:  We can use the `ray.put` function to add the large matrix in the remote object store.
object_ref = ray.put(large_matrix)


# NOTE: We can then use the `ray.get` and the object reference method to retrieve
# the result of remote object from the store.
large_matrix_retrieved = ray.get(object_ref)
assert np.array_equal(large_matrix, large_matrix_retrieved), (
    "The retrieved matrix is not equal to the original one"
)
print("The retrieved matrix is equal to the original one")


# INFO: Pattern: pass an object as a top level argument
# When an object is passed directly as a top-level argument to
# a task, Ray will de-reference the object. This means that Ray
# will fetch the underlying data for all top-level object
# reference arguments, not executing the task until the object
# data becomes fully available.
@ray.remote
def compute(x, y):
    return int(np.matmul(x, y).sum())


mat1_ref = ray.put(np.random.rand(32, 32))
mat2_ref = ray.put(np.random.rand(32, 32))

collection = []
for i in range(10):
    collection.append(compute.remote(mat1_ref, mat2_ref))

results = ray.get(collection)
print(results)


# INFO: Pattern: Passing an object as a nested argument
# When an object is passed within a nested object, for example, within a Python list,
# Ray will not de-reference it. This means that the task will need to call ray.get()
# on the reference to fetch the concrete value. However, if the task never calls ray.get(),
# then the object value never needs to be transferred to the machine the task is running on.
# We recommend passing objects as top-level arguments where possible,
# but nested arguments can be useful for passing objects on to other tasks
# without needing to see the data.
@ray.remote
def echo_and_get(x_list: list[ray.ObjectRef]):
    """This function prints its input values to stdout."""
    print("args:", x_list)
    print("values:", ray.get(x_list))


# NOTE: Put the values (1, 2, 3) into Ray's object store.
a, b, c = ray.put(1), ray.put(2), ray.put(3)

# WARN: Passing an object as a nested argument to `echo_and_get`. Ray does not
# de-reference nested args, so `echo_and_get` sees the references.
echo_and_get.remote([a, b, c])


# INFO: Task Runtime Environments
# When setting up a worker process to run a task, Ray will first prepare the environment for the task.
# This includes things like:
#  - installing dependencies
#  - setting environment variables
# For example, we can set an environment variable:
@ray.remote(
    runtime_env={"env_vars": {"MY_CUSTOM_ENV": "prod"}},
)
def f():
    env = os.environ["MY_CUSTOM_ENV"]
    return f"My custom env is {env}"


runtime_env = {"pip": ["emoji"]}


@ray.remote(runtime_env=runtime_env)
def f2():
    import emoji  # pyright: ignore

    return emoji.emojize("Python is :thumbs_up:")


print(ray.get(f.remote()))
print(ray.get(f2.remote()))
# WARN: pip dependencies in task runtime environments don’t come for free.
# They add to the startup time of the worker process.
# If you find yourself needing to install the same dependencies across many tasks,
# consider baking them into the image you use to start your Ray cluster.
#


# NOTE: On resources requests, available resources, configuring large clusters
# During the scheduling stage, Ray evaluates the resource requirements specified via the
# @ray.remote decorator or within the resources={...} argument. These requirements may include:
#     CPU e.g., @ray.remote(num_cpus=2))
#     GPU e.g., @ray.remote(num_gpus=1))
#     Custom resources: User-defined custom resources like "TPU"
#     Memory
# Ray's scheduler checks the resource specification (sometimes referred to as resource shape) to match tasks and actors with available resources in the cluster. If the exact resource combination is unavailable, Ray may autoscale the cluster.
# You can inspect the current resource availability using:
#
# This returns a dictionary showing the currently available CPUs, GPUs, memory, and any
# custom resources, for example:
ray.available_resources()

# WARN: Pattern: configure the head node to be unavailable for compute tasks.
# When scaling to large clusters, it’s important to ensure that the head node does not handle any compute tasks. Users can indicate that the head node is unavailable for compute by setting its resources:
# resources: {"CPU": 0}


# NOTE: Fractional resources
# Fractional resources allow Ray Tasks to request a fraction of a CPU or GPU (e.g., 0.5), enabling finer-grained resource allocation.
# Let’s consider the above example again:


@ray.remote(num_cpus=0.5)
def remote_add(a, b):
    return a + b


ref = remote_add.remote(2, 3)
print(ref)
ray.get(ref)


# NOTE: Fractional resources include support for multiple accelerators,
# allowing users to load multiple smaller models onto a single GPU.
# This is especially useful for scenarios like batch inference.


@ray.remote
def square(a):
    return a**2


# INFO: Nested Tasks
@ray.remote
def main():
    """
    Ray `DOES NOT` require that all your tasks and their dependencies be arranged
    from one `driver process`
    """
    ref1 = square.remote(1)
    ref2 = square.remote(3)
    # WARN: To avoid deadlocks, Ray yields CPU resources while blocked waiting
    # for a task to complete.
    add_result = remote_add.remote(ref1, ref2)
    return ray.get(add_result)


# NOTE: In this example:
#     1. Our local process requests Ray to schedule a main task in the cluster
#     2. Ray executes the main task in a separate worker process
#     3. Inside main, we invoke multiple expensive_square tasks, which Ray
#        distributes across available worker processes
#     4. Once all “sub tasks” complete, main returns the final value
# This ability for tasks to schedule other tasks using uniform semantics makes Ray particularly powerful and flexible.
print(f"The squared sum result is {ray.get(main.remote())}")


# INFO: Pattern: Pipeline Data Processing and waiting for results
# After launching a number of tasks, you may want to know which ones have finished
# executing without blocking the main process.
# You can use the `ray.wait`
@ray.remote
def expensive_square(x):
    time.sleep(np.random.randint(1, 10))
    return x**2


expensive_compute = []

for i in range(15):
    expensive_compute.append(expensive_square.remote(i))

# NOTE: We can process items as soon as they become available
ready_refs, not_ready_refs = ray.wait(
    expensive_compute
)  # wait for next object ref that is ready

# process new item as soon as it becomes available
while not_ready_refs:
    print(f"{ready_refs[0]} is ready; result: {ray.get(ready_refs[0])}")
    print(f"{len(not_ready_refs)} items not ready... \n")

    ready_refs, not_ready_refs = ray.wait(not_ready_refs)  # wait for next item

    assert len(ready_refs) == 1, (
        f"len(ready_refs) should be 1, got {len(ready_refs)} instead"
    )

print(f"I'm the last item: {ready_refs[0]}; result: {ray.get(ready_refs[0])}")
# Shutdown Ray to free up resources
ray.shutdown()
