import torch
from typing import Any
from torchvision.transforms import Compose, Normalize, ToTensor
import tqdm
from torchvision.models import resnet18
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from ray import tune


def train_pytorch(
    config: dict[str, Any],
) -> (
    None
):  # NOTE: We pass hyperparameters as config. This is typical for Ray Train functions
    # as it will be passed to the Ray Train's Torch Trainer and Ray Tune

    criterion = CrossEntropyLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: torch.nn.Module = build_resnet18().to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"])

    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], drop_last=True, shuffle=True
    )

    for _ in tqdm.tqdm(range(config["num_epochs"]), desc="Epochs", leave=False):
        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        # INFO: We use Ray Tune to report metrics instead of locally reporting
        tune.report({"loss": loss.item()})  # pyright: ignore[reportPossiblyUnboundVariable]


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


tuner = tune.Tuner(
    trainable=tune.with_resources(
        train_pytorch, resources={"gpu": 1}
    ),  # NOTE: We will dedicate 1 GPU to each trial
    param_space={
        "batch_size": tune.grid_search([32, 64, 128]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_epochs": tune.grid_search([1, 2, 3]),
    },
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
    ),
)

results = tuner.fit()
best_trial = results.get_best_result()

print(f"The best config is {best_trial}")
