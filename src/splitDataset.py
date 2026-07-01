from pathlib import Path
import random
import shutil


# =========================
# SPLIT ONE CLASS
# =========================
def split_class_files(
    src_class_dir: Path,
    train_class_dir: Path,
    valid_class_dir: Path,
    test_class_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
):

    if not src_class_dir.exists():
        print(f"Skip missing: {src_class_dir}")
        return

    files = list(src_class_dir.glob("*"))

    if len(files) == 0:
        print(f"Empty folder: {src_class_dir}")
        return

    # FIX: deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(files)

    total = len(files)

    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)

    train_files = files[:train_count]
    valid_files = files[train_count:train_count + valid_count]
    test_files = files[train_count + valid_count:]

    # create dirs safely
    train_class_dir.mkdir(parents=True, exist_ok=True)
    valid_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # copy safely (no overwrite confusion)
    for f in train_files:
        shutil.copy2(f, train_class_dir / f.name)

    for f in valid_files:
        shutil.copy2(f, valid_class_dir / f.name)

    for f in test_files:
        shutil.copy2(f, test_class_dir / f.name)

    print(
        f"{src_class_dir.name}: "
        f"total={total}, train={len(train_files)}, "
        f"valid={len(valid_files)}, test={len(test_files)}"
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    data_dir = Path("./data")
    raw_dir = data_dir / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError("data/raw not found")

    seed = 42
    train_ratio = 0.7
    valid_ratio = 0.15

    class_names = sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])

    # 🔥 HARD RESET (IMPORTANT FIX)
    for split in ["train", "valid", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    # recreate structure clean
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

    print("Done: dataset split completed safely")