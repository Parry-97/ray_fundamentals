from pathlib import Path
import tqdm
import torch
from torchvision.models import resnet18
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def train_loop_torch(
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 2,
    batch_size: int = 32,
    local_path: str = "./checkpoints",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    criterion = CrossEntropyLoss()
    model = load_model_torch().to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
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
        print(f"Epoch {epoch + 1} average loss: {avg_loss}")


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


dataset = MNIST(root="./data", train=True, download=True)
