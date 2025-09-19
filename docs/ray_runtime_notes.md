# Understanding Ray Runtime Environments

This document summarizes key insights about using Ray's `runtime_env` for managing task dependencies.

## 1. Environments are Isolated

The most important concept is **environment isolation**.

- The environment you run your script from (e.g., your project's `.venv`) is the **driver environment**.
- A `runtime_env` specified for a Ray task creates a **separate, isolated environment** on a **worker node**.
- Dependencies installed in a task's `runtime_env` (like `emoji`) **do not** get installed back into your project's driver environment.

## 2. Dependency Installation is Cached

Task dependencies are not installed every single time a task runs.

- When a task with a `runtime_env` runs on a worker for the first time, Ray creates and installs the dependencies.
- This environment is then **cached** on the worker.
- Subsequent tasks with the _exact same_ `runtime_env` will reuse the cached environment, making execution much faster.
- The cache lasts for the lifetime of the worker process.

## 3. `pip` and `conda` are the Default Managers

Ray's `runtime_env` has built-in support for `pip` and `conda`. It does **not** have native support for other package managers like `uv`.

```python
# This works because Ray knows how to use pip
@ray.remote(runtime_env={"pip": ["requests"]})
def my_task():
    import requests
    return requests.get("https://www.google.com").status_code
```

If you need to use a different package manager, the recommended workaround is to use a custom Docker container.

## 4. The `pip` Requirement: Local vs. Multi-Node Clusters

This was a key point of confusion. The reason you might need `pip` in your _local project environment_ is specific to how **local clusters** work.

#### On a Local Cluster (Default Behavior)

- When you run a Ray script on your machine, Ray starts worker processes that are based on the **same environment** as your driver script.
- Therefore, for the `runtime_env` installer to work, it needs to find `pip` within that shared base environment.
- **Conclusion:** If you use `uv` locally, you should still ensure `pip` is installed in that environment (`uv pip install pip`) so Ray's local workers can use it.

#### On a Multi-Node Cluster

- The driver environment (your laptop) and the worker environments (remote servers) are **completely separate**.
- Your local driver environment **does not** need `pip` for `runtime_env` to work on the cluster. It only needs `ray` to connect.
- Each **remote worker node** must be configured to have `pip` in its base Python environment.

## 5. Using Custom Docker Images for Full Control

When you need full control over the execution environment, or if you want to use a custom package manager like `uv`, the best practice is to use a Docker container. This gives you a completely reproducible and portable environment.

The process involves three steps:

### Step 1: Create a `Dockerfile`

A `Dockerfile` is a text file that defines all the commands needed to build your container image. You would place this file in your project root.

**Example `Dockerfile` using `uv`:**

```dockerfile
# Start from a standard Python base image
FROM python:3.11-slim

# Install uv using the recommended installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to the system's PATH
ENV PATH="/root/.local/bin:$PATH"

# Create a virtual environment inside the container (optional but good practice)
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install your dependencies using uv
# You can list them directly or use a requirements.txt / pyproject.toml file
RUN uv pip install emoji "ray[serve]"

# Set the working directory
WORKDIR /app
```

### Step 2: Build the Docker Image

From your terminal, in the same directory as your `Dockerfile`, run the `docker build` command.

```bash
# Build the image and tag it with a memorable name
docker build -t my-ray-app-env:latest .
```

### Step 3: Use the Image in `runtime_env`

In your Python code, point the `runtime_env` to your new image using the `container` key. Ray will then pull and run your task inside a container based on this image, ignoring other keys like `pip`.

```python
@ray.remote(
    runtime_env={"container": {"image": "my-ray-app-env:latest"}}
)
def f_in_container():
    import emoji
    # This code now runs inside your custom container
    return emoji.emojize("Ray is running in a container! :ship:")

# Make sure Docker is running on your machine to execute this
# print(ray.get(f_in_container.remote()))
```

## Summary & Best Practices

- **Be Explicit:** Always define all task dependencies in the `runtime_env`. Do not assume tasks will inherit packages from your project environment.
- **Local Development:** When using a custom package manager like `uv` locally, ensure `pip` is also installed in the environment so Ray's local workers can function correctly.
- **Production/Custom Environments:** For complex dependencies or to use custom package managers like `uv`, the best practice is to build a **custom Docker image** and specify it using the `container` field in `runtime_env`.
