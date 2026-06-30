from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def create_datasets(data_dir: str = "./data"):
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    valid_dir = data_path / "valid"
    test_dir = data_path / "test"

    if not (train_dir.exists() and valid_dir.exists() and test_dir.exists()):
        raise FileNotFoundError("Expected data/train, data/valid and data/test folders.")

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=get_train_transform())
    valid_dataset = datasets.ImageFolder(root=str(valid_dir), transform=get_eval_transforms())
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=get_eval_transforms())

    return train_dataset, valid_dataset, test_dataset

def create_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_dataset, valid_dataset, test_dataset = create_datasets(data_dir=data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader

def get_class_names(data_dir: str = "./data") -> list[str]:
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    if not train_dir.exists():
        raise FileNotFoundError("Expected data/train folder.")
    dataset = datasets.ImageFolder(root=str(train_dir), transform=None)
    return dataset.classes