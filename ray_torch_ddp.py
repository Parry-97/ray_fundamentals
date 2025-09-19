from pathlib import Path
from torch.distributed.fsdp import FullyShardedDataParallel
import tempfile
import torch
from typing import Any
from torchvision.transforms import Compose, Normalize, ToTensor
import tqdm
from torchvision.models import resnet18
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import ray.train
import ray.train.torch


def train_loop_ray_train(
    config: dict[str, Any],
) -> (
    None
):  # NOTE: We pass hyperparameters as config. This is typical for Ray Train functions
    # as it will be passed to the Ray Train's Torch Trainer
    """
    In the case of `Distributed Data Parallel Training`, we can consider having `multiple workers`
    each being responsible for a subset of the data. This is a common pattern in distributed training
    along with a `copy of the model` with synced initial parameters and subsequent `gradient
    synchronization`. The forward, backward pass is therefore duplicated across the cluster.
    By syncing the gradients we make sure the updates are consistent are not applied
    to older versions of the model that may not have incorporated the latest updates.
    Furthermore, syncing gradients is much easier than syncing other components of the
    training loop, such as the optimizer state.
    """

    criterion = CrossEntropyLoss()

    # NOTE: A typical migration roadmap from PyTorch DDP to PyTorrch with Ray Train
    # 1. Configure scale and GPUs
    # 2. Migrate the model to Ray Train
    # 3. Migrate the dataset to Ray Train
    # 4. Build checkpoints and metrics reporting
    # 5. Configure persistent storage

    model = load_model_ray_train()  # NOTE: We use Ray Train to wrap the model with DDP
    optimizer = Adam(model.parameters(), lr=1e-5)

    # Calculate the batch size for each worker
    global_batch_size = int(config["batch_size"])
    world_size = ray.train.get_context().get_world_size()
    batch_size = global_batch_size // world_size

    # NOTE: We use Ray Train to wrap the dataloader  as a DistributedSampler
    dataloader = build_data_loader_ray_train(batch_size=batch_size)

    for epoch in tqdm.tqdm(range(config["num_epochs"]), desc="Epochs", leave=False):
        # NOTE: This set the epoch for the DistributedSampler.
        # This ensures all replicas use a different random ordering
        # for each epoch. Otherwise, the next iteration of this
        # sampler will yield the same ordering.

        # WARN: This works only if we are using a DistributedSampler and more than one worker
        if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0

        for images, labels in tqdm.tqdm(dataloader):
            # NOTE: We don't manually move images and labels to a
            # device like "cuda", as this is automatically done by Ray Train
            # dataloader
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # INFO: We use Ray Train to report metrics instead of locally reporting
        metrics = report_metrics_ray_train(loss=epoch_loss, epoch=epoch)

        # INFO: We use Ray Train to save checkpoints and metrics
        save_checkpoint_and_metrics_ray_train(model, metrics)


# INFO: We now can configure scale and GPUs with a `ScalingConfig`
scaling_config = ray.train.ScalingConfig(
    num_workers=1,  # The number of worker processes
    use_gpu=False,
)

# INFO: Ray Train is built around four key concepts:
# 1. Training Function: A python function that contains your model training logic
# 2. Worker: A process that runs the training function
# 3. Scaling Config: A configuration that specifies the number of workers and compute resources (CPU, TPU, GPU)
# 4. Trainer: A python class (Ray Actor) that ties together the training function, workers,
#    and scaling config to execute a distributed training job.


def build_resnet18() -> torch.nn.Module:
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    return model


def load_model_ray_train() -> torch.nn.Module:
    """
    Migrate the model to Ray Train. To do so we can use the `prepare_model` utility function
    to automatically move the model to the correct device and wrap the model in
    PyTorch's DistributedDataParallel
    or FullyShardedDataParallel.
    """
    model: torch.nn.Module = build_resnet18()
    # NOTE: We can provide the prepare_model with additional parameters such as:
    # - parallel_strategy: "ddp" (default) or "fsdp"
    # - parallel_strategy_kwargs: to pass additional parameters to "ddp" or "fsdp"
    model = ray.train.torch.prepare_model(model)
    return model


def build_data_loader_ray_train(
    batch_size: int = 32,
) -> torch.utils.data.DataLoader[torch.Tensor]:
    """
    We can also use the `prepare_dataloader` utility function to automatically:
    1. Move the batches to the right device
    2. Copy data from the CPU to the GPU
    3. Wrap the dataloader in a `DistributedSampler`, if using more than one worker.
       Each worker will be responsible for an exclusive subset of the data.
    This utility function allows users to use the same exact code regardless of the number of workers.
    or device type being used (CPU, GPU).
    """
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )

    # WARN: This step isn't necessary if we are integrating the Ray Train workload with Ray Data.
    # It's especially useful if preprocessing is CPU-heavy and the user wants to run preprocessing
    # and training on separate instances.

    # NOTE: The default behavior is to automatically add a DistributedSampler, move to device, and copy data
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    return train_loader


# NOTE: Reporting metrics and checkpoints
def report_metrics_ray_train(loss: float, epoch: int) -> dict:
    """
    To monitor progress, we can continue to print/log metrics as before. This time we choose to log from
    all workers, as we can use the `get_world_rank` utility function to get the rank of the current worker.
    """
    metrics = {"loss": loss, "epoch": epoch}
    world_rank = ray.train.get_context().get_world_rank()
    print(f"{metrics=} {world_rank=}")
    return metrics


def save_checkpoint_and_metrics_ray_train(
    model: torch.nn.Module, metrics: dict
) -> None:
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # WARN: The model is actually wrapped in a DistributedDataParallel, so we need to access the module
        # Both DistributedDataParallel and FullyShardedDataParallel have a `module` attribute
        # and they both subclass torch.nn.Module, hence the return type from `prepare_model`
        # and the `model` parameter are of type torch.nn.Module
        # This will only work in a distributed setting where more than 1 worker is used
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(
            model, FullyShardedDataParallel
        ):
            torch.save(
                model.module.state_dict(), Path(temp_checkpoint_dir) / "model.pth"
            )
        else:
            torch.save(model.state_dict(), Path(temp_checkpoint_dir) / "model.pth")  # pyright: ignore[reportAttributeAccessIssue]

        # WARN: Only the metrics reported by the rank 0 worker will be attached to the checkpoint.
        ray.train.report(
            metrics, checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        )


# NOTE: Checkpoint Lifecycle
# Given it is the the same model across the workers since they are synced by the `all_reduce` algorithm,
# we can instead only build the checkpoint on the 0 rank worker. We will still need to
# call `report` to ensure that the training loop is synced.
# Ray Traing expects all workers to be able to write files to the same persistent storage location


storage_folder = "./cluster_storage"
storage_path = f"file://{Path(storage_folder).resolve()}/training/"
# NOTE: We use RunConfig object to specicy the path where results (checkpoints and artifacts ) will be saved
run_config = ray.train.RunConfig(
    storage_path=storage_path, name="distributed-mnist-resnet"
)

# INFO: We can now launch the distributed training loop run using the `TorchTrainer` class
trainer = ray.train.torch.TorchTrainer(
    run_config=run_config,
    train_loop_config={"batch_size": 128, "num_epochs": 1},
    train_loop_per_worker=train_loop_ray_train,
)

# INFO: Calling `trainer.fit()` will start the distributed training loop
# After training loop is completed, a `Result` object is returned which contains
# information about the training run, including metrics and checkpoints reported
# during the training run.
result = trainer.fit()
