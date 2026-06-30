from pathlib import Path
import torch
from PIL import Image
from torchvision import models

from src.dataset import get_class_names, get_eval_transforms

def load_model(weights_path: str, num_classes: int, device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path: str, weights_path: str = "./artifacts/best_model.pt", data_dir: str = "./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = get_class_names(data_dir=data_dir)
    model = load_model(weights_path=weights_path, num_classes=len(class_names), device=device)

    image = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    print(f"Prediction: {class_names[pred_idx]}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict class for one image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="./artifacts/best_model.pt", help="Path to model weights")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to dataset root")
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    predict_image(image_path=args.image, weights_path=args.weights, data_dir=args.data_dir)