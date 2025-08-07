import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from mae.model import MAE

# ==== Configuration ====
IMG_SIZE = 224
NUM_CLASSES = 2  # à adapter
FREEZE_ENCODER = False  # ou True
PRETRAINED_MAE_PATH = 'mae_pretrained.pth'

# ==== Dataset ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(root='data/train', transform=transform)
val_dataset = ImageFolder(root='data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ==== Classification model ====
class MAEForClassification(nn.Module):
    def __init__(self, pretrained_mae, num_classes=2, freeze_encoder=False):
        super().__init__()
        self.patch_embed = pretrained_mae.patch_embed
        self.encoder = pretrained_mae.encoder
        self.pos_embed = pretrained_mae.pos_embed

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)  # (B, N, D)
        x = x.mean(dim=1)    # global average pooling
        return self.classifier(x)

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained MAE
pretrained_mae = MAE()
pretrained_mae.load_state_dict(torch.load(PRETRAINED_MAE_PATH, map_location=device))
pretrained_mae.to(device)

model = MAEForClassification(pretrained_mae, num_classes=NUM_CLASSES, freeze_encoder=FREEZE_ENCODER)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# ==== Training ====
for epoch in range(10):
    model.train()
    train_loss = 0
    correct = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    acc = correct / len(train_dataset)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Accuracy = {acc:.4f}")

    # Optional validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
    val_acc = val_correct / len(val_dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")
