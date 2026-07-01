"""
Streamlit веб-приложение для классификации Тигр vs Лев
"""

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from pathlib import Path
import json


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Тигр vs Лев",
    page_icon="🐯",
    layout="centered",
)


# =========================
# STYLE
# =========================
st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 2.3em;
        font-weight: bold;
        color: #FF6B35;
    }
    .result-container {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .tiger {
        background-color: #fff3e0;
        border: 3px solid #FF6B35;
    }
    .lion {
        background-color: #fce4ec;
        border: 3px solid #E91E63;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">🐯 Тигр или Лев?</div>', unsafe_allow_html=True)


# =========================
# LOAD CLASS MAP (FIXED)
# =========================
@st.cache_resource
def load_classes(path="artifacts/classes.json"):
    if not Path(path).exists():
        raise FileNotFoundError("classes.json not found. Train model first.")

    with open(path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # 🔥 гарантируем правильный порядок
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return idx_to_class


idx_to_class = load_classes()


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model(model_path="./artifacts/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)   # 🔥 FIX: строго same as training
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    if not Path(model_path).exists():
        st.warning("Модель не найдена — будет случайный результат")

    else:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        st.success("Модель загружена")

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()


# =========================
# TRANSFORM
# =========================
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    return transform(image).unsqueeze(0)


# =========================
# PREDICT
# =========================
def predict(image_tensor):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        out = model(image_tensor)
        probs = torch.softmax(out, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    return pred, conf


# =========================
# UI
# =========================
uploaded = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Изображение", use_container_width=True)

    tensor = preprocess(img)
    pred, conf = predict(tensor)

    class_name = idx_to_class[pred]

    ru_map = {
        "lion": "Лев",
        "tiger": "Тигр"
    }

    predicted_ru = ru_map[class_name]

    emoji = "🐯" if class_name == "tiger" else "🦁"
    color_class = class_name

    st.markdown(f"""
    <div class="result-container {color_class}">
        <div style="font-size:40px">{emoji}</div>
        <h2>{predicted_ru}</h2>
        <p>Уверенность: {conf*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Загрузите изображение чтобы начать")