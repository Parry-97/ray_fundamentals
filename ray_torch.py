from pathlib import Path
import torch
import datetime
from torchvision.datasets.celeba import csv
from torchvision.transforms import Compose, Normalize, ToTensor
import tqdm
from torchvision.models import resnet18
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def train_loop_torch(
    num_epochs: int = 2,
    batch_size: int = 32,
    local_path: str = "./checkpoints",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    criterion = CrossEntropyLoss()
    model = load_model_torch().to(device)
    dataloader = build_data_loader_torch(batch_size=batch_size)
    optimizer = Adam(model.parameters(), lr=1e-5)

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
        metrics = report_metrics_torch(loss=avg_loss, epoch=epoch)
        Path(local_path).mkdir(parents=True, exist_ok=True)
        save_checkpoint_and_metrics_torch(
            model=model, metrics=metrics, local_path=local_path
        )


def load_model_torch() -> torch.nn.Module:
    model = build_resnet18()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


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


def build_data_loader_torch(batch_size: int = 32) -> torch.utils.data.DataLoader:
    # NOTE: Composes several transforms together.
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


def report_metrics_torch(loss: float, epoch: int) -> dict:
    metrics = {"loss": loss, "epoch": epoch}
    print(f"\nEpoch {epoch} average loss: {loss}")
    return metrics


def save_checkpoint_and_metrics_torch(
    model: torch.nn.Module, metrics: dict, local_path: str
) -> None:
    with open(Path(local_path) / "metrics.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.values())
    checkpoint_path = Path(local_path) / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
storage_folder = "./cluster_storage"
local_path = f"{storage_folder}/torch-{timestamp}/"

train_loop_torch(num_epochs=1, local_path=local_path)
