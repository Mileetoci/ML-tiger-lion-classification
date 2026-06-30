from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torch.optim.lr_scheduler import StepLR

from src.dataset import create_dataloaders, get_class_names

def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

def evaluate_test(model, loader, criterion, device):
    return eval_one_epoch(model, loader, criterion, device)

def train_model(data_dir="./data", epochs=10, batch_size=16, lr=1e-3, model_path="./artifacts/best_model.pt", num_workers=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = get_class_names(data_dir)
    print("Classes:", class_names)
    print("Device:", device)

    train_loader, valid_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    best_valid_acc = 0.0
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = eval_one_epoch(model, valid_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model: {model_path}")

        scheduler.step()

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))

    test_loss, test_acc = evaluate_test(model, test_loader, criterion, device)
    print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f}")

if __name__ == "__main__":
    train_model()