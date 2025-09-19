# Ray `num_cpus` Explained

## What does `num_cpus` mean?

- `num_cpus` in Ray refers to **CPU logical units**, not strictly physical cores.
- Ray detects the number of **logical CPUs** on the node (e.g., with hyperthreading: 8 physical cores → 16 logical CPUs).
- When you set `@ray.remote(num_cpus=N)`, Ray reserves **N logical CPU slots** for that task or actor.

## Example

- Machine: 8 physical cores with hyperthreading → 16 logical CPUs.
- `num_cpus=1` → reserves 1 logical CPU (≈ half a physical core).
- To approximate a full physical core in this setup → `num_cpus=2`.

## Key Points

- Ray only does **scheduling**, not OS-level pinning.  
  The actual CPU assignment is left to the operating system scheduler.
- For **true CPU affinity** (pinning tasks to cores), you must configure it manually (e.g., with `taskset`).

## Related

- `num_gpus` in Ray reserves full GPU devices (not fractional by default).

> You can verify the resources available on a node with psutil:
>
> ```python
>   print(psutil.cpu_count(logical=True))
> ```
