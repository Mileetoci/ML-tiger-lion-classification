from pathlib import Path
import torch
from PIL import Image
from torchvision import models
import json

from src.dataset import get_eval_transforms


# =========================
# LOAD CLASS MAP (FIX 100%)
# =========================
def load_class_mapping(path="artifacts/classes.json"):
    if not Path(path).exists():
        raise FileNotFoundError("classes.json not found. Train model first.")

    with open(path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


# =========================
# MODEL
# =========================
def load_model(weights_path: str, num_classes: int, device: torch.device):
    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model


# =========================
# PREDICT
# =========================
def predict_image(
    image_path: str,
    weights_path: str = "./artifacts/best_model.pt",
    data_dir: str = "./data",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 FIX: стабильный mapping (НЕ ImageFolder!)
    idx_to_class = load_class_mapping()

    model = load_model(
        weights_path=weights_path,
        num_classes=len(idx_to_class),
        device=device
    )

    image = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    predicted_class = idx_to_class[pred_idx]

    # RU labels
    ru_map = {
        "tiger": "Тигр",
        "lion": "Лев"
    }

    print("\n===== PREDICTION =====")
    print(f"Class: {ru_map.get(predicted_class, predicted_class)}")
    print(f"Confidence: {confidence:.4f}")
    print("======================\n")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict image class")

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default="./artifacts/best_model.pt")

    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    predict_image(
        image_path=args.image,
        weights_path=args.weights
    )