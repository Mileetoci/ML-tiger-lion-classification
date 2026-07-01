from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torch.optim.lr_scheduler import StepLR

from src.dataset import create_dataloaders, get_class_names

def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Build ResNet18 model for classification.
    
    Args:
        num_classes: Number of output classes
        freeze_backbone: If True, freeze backbone weights (transfer learning)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if freeze_backbone:
        # Freeze all backbone parameters
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
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
    """Evaluate for one epoch."""
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
    """Evaluate on test set."""
    return eval_one_epoch(model, loader, criterion, device)

def train_model(
    data_dir="./data",
    epochs=10,
    batch_size=16,
    lr=1e-3,
    model_path="./artifacts/best_model.pt",
    num_workers=2,
    freeze_backbone=True,
):
    """
    Train the classification model.
    
    Args:
        data_dir: Path to dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        model_path: Path to save best model weights
        num_workers: Number of workers for data loading
        freeze_backbone: Whether to freeze ResNet backbone weights
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = get_class_names(data_dir)
    print("Classes:", class_names)
    print("Device:", device)
    print(f"Freeze backbone: {freeze_backbone}")

    train_loader, valid_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(num_classes=len(class_names), freeze_backbone=freeze_backbone).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimize only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
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
    print(f"\n{'='*50}")
    print(f"Test Results | loss={test_loss:.4f} acc={test_acc:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    train_model()
