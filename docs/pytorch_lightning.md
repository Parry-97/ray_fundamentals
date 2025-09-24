# PyTorch vs PyTorch Lightning

## 1. PyTorch (Regular)

- **What it is:**  
  A deep learning framework providing the building blocks for defining, training, and deploying neural networks.

- **Level of abstraction:**  
  Low to mid-level → fine-grained control.

- **You implement manually:**
  - Model definition (`nn.Module`)
  - Training loop (forward, backward, optimizer step, loss computation)
  - Validation/test loops
  - Device placement (CPU/GPU)
  - Logging/metrics
  - Checkpointing/resuming

- **Best for:**  
  Research, custom training methods, when full flexibility is needed.

---

## 2. PyTorch Lightning

- **What it is:**  
  A higher-level framework built **on top of PyTorch**, enforcing structure and reducing boilerplate.

- **Level of abstraction:**  
  High-level → you focus on model logic, Lightning handles the training loop.

- **You implement manually:**
  - Model logic in `LightningModule`
  - `training_step`, `validation_step`, `test_step`
  - Optimizers/schedulers in `configure_optimizers`

- **Lightning handles automatically:**
  - Training/validation/testing loops
  - Device placement (CPU, GPU, TPU, multi-GPU)
  - Mixed precision (AMP)
  - Logging integrations (TensorBoard, WandB, etc.)
  - Checkpointing, early stopping, resuming

- **Best for:**  
  Production, standardized workflows, collaboration, rapid prototyping.

---

## 3. Example

### PyTorch (manual training loop)

```python
import torch
from torch import nn, optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc(x)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in dataloader:
        x = x.view(x.size(0), -1)
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### PyTorch Lightning (abstracted loop)

```python
import torch
from torch import nn
import pytorch_lightning as pl

class LitNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(LitNet(), train_dataloader)
```

---

## 4. Key Differences

| Aspect                  | PyTorch (Regular)         | PyTorch Lightning       |
| ----------------------- | ------------------------- | ----------------------- |
| **Control**             | Full (manual)             | Abstracted loops        |
| **Code size**           | Verbose                   | Concise                 |
| **Flexibility**         | Maximum                   | High (with structure)   |
| **Scaling (multi-GPU)** | Manual setup              | Built-in support        |
| **Logging/Checkpoints** | Manual                    | Built-in                |
| **Best for**            | Research, custom training | Production, prototyping |

---

### TL;DR

- **PyTorch = toolbox** → you build everything yourself.
- **PyTorch Lightning = framework** → organizes training, adds batteries (logging, scaling, checkpointing).
