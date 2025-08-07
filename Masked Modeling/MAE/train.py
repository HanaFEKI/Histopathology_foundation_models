import torch
import torch.nn as nn
import torch.optim as optim
from mae.model import MAE
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        preds, targets, mask = model(imgs)

        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")
