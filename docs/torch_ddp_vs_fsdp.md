# **DDP vs FSDP in PyTorch**

## 1. **Distributed Data Parallel (DDP)**

- **What it is**
  The standard way to scale PyTorch training across multiple GPUs/nodes.
- **How it works**
  - Each worker (GPU) has a **full copy of the model parameters**.
  - Each worker processes a different shard of data (data parallelism).
  - After every forward/backward pass:
    - Gradients are synchronized across workers (`all_reduce`).
    - All replicas update weights identically.

- **Pros**
  - Simple, robust, widely used.
  - Good performance up to moderate model sizes.

- **Cons**
  - **Model replication**: every worker stores the _entire model_.
  - Memory bottleneck for **very large models** (billions of parameters).

---

## 2. **Fully Sharded Data Parallel (FSDP)**

- **What it is**
  A newer approach (inspired by ZeRO from DeepSpeed) to train **very large models** by sharding model states.
- **How it works**
  - Instead of each worker storing a full model copy, FSDP **shards model parameters, gradients, and optimizer states across workers**.
  - Each worker only holds a **slice** of the parameters at a time.
  - During forward/backward:
    - Parameters are gathered _just-in-time_ for computation.
    - After use, parameters are freed (released from memory).
    - Gradients are also sharded across workers.

- **Pros**
  - Much lower memory footprint (enables training models with **trillions** of parameters).
  - Efficient for extreme-scale distributed training.

- **Cons**
  - More communication overhead (frequent gather/scatter).
  - More complex to configure and tune.
  - Sometimes slower than DDP for small/medium models.

---

## 3. **Quick Comparison**

| Feature           | DDP 🚀                                                    | FSDP 🪶                                                                     |
| ----------------- | --------------------------------------------------------- | --------------------------------------------------------------------------- |
| Model replication | Full copy on each worker                                  | Sharded across workers                                                      |
| Gradient sync     | AllReduce full grads                                      | Sharded grads                                                               |
| Memory efficiency | Moderate (needs full model)                               | High (shards parameters & states)                                           |
| Best for          | Small/medium models (up to a few hundred million params)  | Very large models (hundreds of billions+)                                   |
| Complexity        | Easy to use (`torch.nn.parallel.DistributedDataParallel`) | More complex API/config (`torch.distributed.fsdp.FullyShardedDataParallel`) |

---

✅ **Rule of thumb**:

- Use **DDP** unless you hit **out-of-memory** due to model size.
- Switch to **FSDP** (or DeepSpeed ZeRO) when scaling to huge models.

---
