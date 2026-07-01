from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import json

from src.dataset import create_dataloaders, get_class_names


# =========================
# MODEL
# =========================
def build_model(num_classes=2, freeze_backbone=True):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):

    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


# =========================
# TRAIN LOOP
# =========================
def train_model(
    data_dir="./data",
    epochs=10,
    batch_size=16,
    lr=1e-3,
    model_path="./artifacts/best_model.pt",
    num_workers=2,
    freeze_backbone=True,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 FIX: берём порядок ИЗ ImageFolder (это 100% источник истины)
    class_names = get_class_names(data_dir)
    class_names = sorted(class_names)

    print("FINAL CLASS ORDER:", class_names)

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    Path("artifacts").mkdir(exist_ok=True)

    with open("artifacts/classes.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    train_loader, valid_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(len(class_names), freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    best_acc = 0.0
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, valid_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print("Saved best model:", model_path)

    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)

    print("\nTEST:")
    print(f"loss={test_loss:.4f} acc={test_acc:.4f}")