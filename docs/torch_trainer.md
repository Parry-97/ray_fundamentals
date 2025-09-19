# Ray Train — TorchTrainer and Worker Mapping

## 1. TorchTrainer vs Ray Actors

- **Ray Actor**  
  A stateful worker process created and managed by Ray.  
  Lives on a Ray node and persists state across method calls.

- **TorchTrainer**  
  Not a Ray Actor itself.  
  It is a _driver/manager_ object that orchestrates distributed training.  
  Under the hood, it **launches multiple Ray Actors** (workers) according to your `ScalingConfig`.

**Key Point**:  
`TorchTrainer ≠ one Ray Actor`
Instead: `TorchTrainer → many Ray Actor workers`.

---

## 2. Worker Mapping

Example:

```python
trainer = TorchTrainer(
    train_loop_per_worker=...,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
)
```

- `num_workers=4` → Ray spawns **4 worker actors**.
- Each actor is allocated resources (CPU cores, GPUs) as specified in `ScalingConfig`.
- Each actor runs `train_loop_per_worker`.

---

## 3. How Load is Distributed

- **Data Parallelism (DDP)**
  - Each worker gets a shard of the dataset.
  - Workers each process mini-batches from their shard.
  - Gradients are synchronized (e.g. via `torch.distributed.all_reduce`).
  - Every worker keeps a full copy of the model weights.

- **Load Balancing**
  - Ray Train does **static sharding** of data.
  - No dynamic work stealing per batch.
  - All workers handle roughly equal amounts of data.

---

## 4. End-to-End Flow

1. Call `trainer.fit()`.
2. TorchTrainer spawns `num_workers` Ray Actors.
3. Each actor runs `train_loop_per_worker`.
4. Ray sets up the PyTorch distributed backend (DDP).
5. Training happens in parallel:
   - Each worker processes its dataset shard.
   - Gradients are synchronized across workers.
   - Weights are updated consistently.

---

## 5. Key Takeaways

- TorchTrainer is **not** a single actor, but **a controller of multiple worker actors**.
- Distribution strategy = **Data Parallelism (DDP)**.
- Each worker runs the training loop independently on its shard of data, then synchronizes gradients.
- Load distribution is static (equal shard sizes), not dynamic.
