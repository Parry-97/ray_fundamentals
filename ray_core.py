import ray


def add(a, b):
    return a + b


@ray.remote
def remote_add(a, b):
    """
    Remote Functions are also called Tasks in Ray.
    Basically they are stateless functions that can be run in parallel.
    They complement stateful actors, which instead correspond to classes.
    """
    return a + b


def main():
    print("Hello from ray-fundamentals!")
    # Native Python function are invoked by simply calling them
    print(add(1, 2))
    # NOTE: Remote Ray functions are executed as tasks by
    # calling them with `.remote()` suffix.
    #  1. Ray schedules the function execution as a task in a separate process in the cluster
    #  2. Ray IMMEDIATELY returns an ObjectRef (a reference to a future result)
    #  3. The cluster executes the task in the background
    ref = remote_add.remote(1, 2)
    print(ref)

    # NOTE: If we want to wait (block) and retrieve the corresponding object, we can use ray.get
    print(ray.get(ref))

    # INFO: ray.get is a blocking call, so it will wait until the result is ready
    # and can accept as input a list of object references to process them in parallel
    # as batch rather than sequentially
    ref2, ref3 = remote_add.remote(1, 2), remote_add.remote(2, 3)
    print(ray.get([ref2, ref3]))

    results = []
    # WARN: This is a common antipattern in Ray since each call to
    # get() blocks until the result is ready.
    for item in range(4):
        output = ray.get(remote_add.remote(item, item))
        results.append(output)

    print(results)

    # INFO: A better solution would be as follows

    out_refs = []
    for item in range(4):
        obj_ref = remote_add.remote(item, item)
        out_refs.append(obj_ref)

    # NOTE: Doing so all remote calls are scheduled and executed in parallel
    results = ray.get(out_refs)
    print(results)


if __name__ == "__main__":
    main()
