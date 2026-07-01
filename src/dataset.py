from pathlib import Path
from typing import List, Tuple, Dict
import json

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# =========================
# TRANSFORMS
# =========================
def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =========================
# SAVE CLASS MAPPING 🔥 FIX
# =========================
def save_class_mapping(class_to_idx: Dict[str, int], save_path="artifacts/classes.json"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=4, ensure_ascii=False)


# =========================
# DATASETS
# =========================
def create_datasets(data_dir: str = "./data") -> Tuple[Dataset, Dataset, Dataset]:
    data_path = Path(data_dir)

    train_dir = data_path / "train"
    valid_dir = data_path / "valid"
    test_dir = data_path / "test"

    missing = [d.name for d in [train_dir, valid_dir, test_dir] if not d.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing folders: {missing}. Required: data/train, data/valid, data/test"
        )

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_train_transform()
    )

    valid_dataset = datasets.ImageFolder(
        root=str(valid_dir),
        transform=get_eval_transforms()
    )

    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=get_eval_transforms()
    )

    # 🔥 FIX: сохраняем ЕДИНЫЙ порядок классов
    save_class_mapping(train_dataset.class_to_idx)

    return train_dataset, valid_dataset, test_dataset


# =========================
# DATALOADERS
# =========================
def create_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset, valid_dataset, test_dataset = create_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader


# =========================
# CLASS NAMES (SAFE VERSION)
# =========================
def get_class_names(data_dir: str = "./data") -> List[str]:
    train_dir = Path(data_dir) / "train"

    if not train_dir.exists():
        raise FileNotFoundError("Missing data/train folder")

    dataset = datasets.ImageFolder(root=str(train_dir))

    return list(dataset.classes)
