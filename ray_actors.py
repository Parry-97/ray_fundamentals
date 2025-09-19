import os
import shutil
import ray

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


# INFO:  Actors extend the Ray API from functions (tasks) to classes.
# An actor is a stateful worker.
#
# WARN: When a new actor is instantiated, A NEW
# WORKER IS CREATED, and methods of the actor are scheduled on that specific
# worker and can access and mutate the state of that worker.
#
# NOTE: Similarly to Ray Tasks, actors support CPU and GPU compute as well as fractional resources.
# Let’s look at an example of an actor which maintains a running balance.


# INFO: We can define an actor with the @ray.remote decorator and
@ray.remote
class Accounting:
    """
    Actor is a remote, stateful Python class.
    The most common use case for actors is with state that is not mutated
    but is large enough that we may want to load it only once and ensure
    we can route calls to it over time, such as a large AI model.
    """

    def __init__(self):
        self.total = 0

    def add(self, amount):
        self.total += amount

    def remove(self, amount):
        self.total -= amount

    def get_total(self):
        return self.total


# NOTE: We can use <class_name>.remote() to ask Ray to construct an instance
# of this actor somewhere in the cluster.
# We get an actor handle which we can use to communicate with that actor,
# pass to other code, tasks, or actors, etc
acc = Accounting.remote()

# NOTE: We can send a message to an actor by calling a method on the actor handle
# using RPC semantics and we get back an ObjectRef
print(acc.get_total.remote())  # pyright: ignore[reportAttributeAccessIssue]


print(f"The current balance is {ray.get(acc.get_total.remote())}")  # pyright: ignore[reportAttributeAccessIssue]

# NOTE: We can mutate the state of the actor by calling a method on the actor handle
acc.add.remote(100)  # pyright: ignore[reportAttributeAccessIssue]
acc.remove.remote(10)  # pyright: ignore[reportAttributeAccessIssue]

print(f"The current balance is {ray.get(acc.get_total.remote())}")  # pyright: ignore[reportAttributeAccessIssue]


@ray.remote
class LinearModel:
    def __init__(self, w0: float, w1: float):
        self.w0 = w0
        self.w1 = w1

    def convert(self, celsius: float):
        return self.w0 * celsius + self.w1


fahrenheit_converter = LinearModel.remote(9 / 5, 32)
print(
    f"The current temperature is {ray.get(fahrenheit_converter.convert.remote(0))} degrees Fahrenheit"  # pyright: ignore[reportAttributeAccessIssue]
)
