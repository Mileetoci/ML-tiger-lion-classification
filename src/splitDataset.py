from pathlib import Path
import random
import shutil

def split_class_files(
    src_class_dir: Path,
    train_class_dir: Path,
    valid_class_dir: Path,
    test_class_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
):
    files = [p for p in src_class_dir.iterdir() if p.is_file()]
    files.sort()
    random.Random(seed).shuffle(files)

    total = len(files)
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)
    test_count = total - train_count - valid_count

    train_files = files[:train_count]
    valid_files = files[train_count : train_count + valid_count]
    test_files = files[train_count + valid_count :]

    for f in train_files:
        shutil.copy2(f, train_class_dir / f.name)
    for f in valid_files:
        shutil.copy2(f, valid_class_dir / f.name)
    for f in test_files:
        shutil.copy2(f, test_class_dir / f.name)

    print(f"{src_class_dir.name}: total={total}, train={len(train_files)}, valid={len(valid_files)}, test={len(test_files)}")

if __name__ == "__main__":
    data_dir = Path("./data")
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError("data/raw not found")

    seed = 42
    train_ratio = 0.7
    valid_ratio = 0.15

    class_names = [p.name for p in raw_dir.iterdir() if p.is_dir()]
    class_names.sort()

    for split_name in ["train", "valid", "test"]:
        for class_name in class_names:
            class_dir = data_dir / split_name / class_name
            if class_dir.exists():
                shutil.rmtree(class_dir)
            class_dir.mkdir(parents=True)

    for class_name in class_names:
        split_class_files(
            src_class_dir=raw_dir / class_name,
            train_class_dir=data_dir / "train" / class_name,
            valid_class_dir=data_dir / "valid" / class_name,
            test_class_dir=data_dir / "test" / class_name,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
        )
    print("Done: raw dataset copied into train/valid/test")