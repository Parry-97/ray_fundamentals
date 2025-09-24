from typing import Any

import numpy as np
import torch
import tqdm
from ray.tune import schedulers
from ray import tune
from ray.tune import search
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

# INFO: Loading and visualizing the data


def build_data_loader_torch(batch_size: int = 32) -> torch.utils.data.DataLoader:
    """
    We are going to work again with the MNIST dataset.
    This consists of 60,000 training images and 10,000 test images.
    Each image is a 28x28 grayscale image.
    """

    # NOTE: Composes several transforms together, in this case, a ToTensor and a Normalize.
    # For Normalize, we are normalizing the data to have a mean of 0.5 and a standard deviation of 0.5.
    # We supplied a single for each tuple since it is a single grayscale image.
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader


def train_loop_torch(num_epochs: int = 2, batch_size: int = 128, lr: float = 1e-5):
    """
    We'll start with a basic PyTorch implementation to establish a baseline
    before moving to advanced techniques. This will give us a good foundation
    for understanding the benefits of Hyperparameter Tuning and distributed
    training in later sections.

    The parameters we provided are the ones we would like to tune.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # INFO: We define the loss function
    criterion = CrossEntropyLoss()

    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    # NOTE: We are manually moving the model to the device
    model.to(device)

    dataloader = build_data_loader_torch(batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs", leave=False):
        epoch_loss = 0
        for images, labels in tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch} average loss: {avg_loss}")


# train_loop_torch()

# NOTE: Let's now see if we can tune the Hyperparameters of our model to get a better loss.
# But Hyperparameter tuning is an expensive process and I will take a long time to run
# sequentially.
# Ray Tune is a distributed Hyperparameter Tuning library that can help us speed up the
# process.

# INFO: Ray Tune is a Python library for experiment execution and
# distributed hyperparameter optimization, built on top of Ray.
# This means we can carry out hyperparameter optimization at scale.
# We can scale out hyperparameter optimization by running it on multiple machines.

# INFO: It connects ML frameworks like PyTorch, TensorFlow, and Keras hyperparameter optimization tools
# like Optuna or HyperOpt


def simple_model(
    distance: np.ndarray,
    a: float,
) -> np.ndarray:  # NOTE: `a` is our hyperparameter to optimize
    """Let's get started with a fairly simple example first."""
    return distance * a


def train_loop_simple_model(
    config: dict[str, Any],
) -> None:
    # WARN: This is the typical signature of a Ray Tune Trainable
    # It needs to accept a `config` argument. This because Ray Tune will pass
    # the hyperparameters to this function as a dictionary.
    distances = np.random.rand(100)
    total_amts = distances * 10

    a = config["a"]
    predictions = simple_model(distances, a)
    rmse = np.sqrt(np.mean((predictions - total_amts) ** 2))

    tune.report({"rmse": rmse})  # NOTE: This is how we report metrics back to Ray Tune


# INFO: Next we define and run the hyperparameter tuning job by following these steps:
# 1. Create a Tuner object
# 2. Setup the Tuner configuration with param_space and tune_config
# 3. Call the Tuner.fit() and get the results
# 4. Get the best result

tuner = tune.Tuner(
    trainable=train_loop_simple_model,  # NOTE: We pass the train_loop_simple_model function as the trainable
    param_space={
        "a": tune.randint(
            0, 20
        )  # NOTE: We define the parameter space for the hyperparameters
    },
    tune_config=tune.TuneConfig(
        # We define the metric we want to optimize
        metric="rmse",  # WARN: This metric should be reported with tune.report()
        mode="min",
        num_samples=5,  # Number of times to sample from the hyperparameter space
        # NOTE: This is the default scheduler, no early stopping, just runs all trials as they are submitted
        scheduler=schedulers.FIFOScheduler(),
        # NOTE: Default search algorithm, combination of random search and grid search (depending on the
        # size of the search space)
        # We can also use a more complicated search algorithm that are specified through integrations
        # such as Optuna or HyperOpt
        search_alg=search.BasicVariantGenerator(),
    ),
)
results = tuner.fit()
best_config = results.get_best_result()

print(f"The best config is: {best_config}")
