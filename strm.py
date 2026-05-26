# app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
import numpy as np
import io

st.set_page_config(
    page_title="Определение местоположения",
    page_icon="📍",
    layout="centered"
)

# ─── Model yuklash ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    data = torch.load('geo_model_final.pth', map_location='cpu', weights_only=False)

    class GeoClassifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            net = tvm.efficientnet_b3(weights=None)
            self.features = net.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1536, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            return self.head(x)

    model = GeoClassifier(data['num_classes'])
    model.load_state_dict(data['model_state'])
    model.eval()
    return model, data['label_to_coords']

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def predict(image_bytes, model, label_to_coords, top_k=3):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    topk_vals, topk_idx = probs.topk(top_k)
    weights = topk_vals.numpy()
    weights = weights / weights.sum()
    idx = topk_idx.numpy()
    coords = label_to_coords.loc[idx]
    pred_lat = (weights * coords['latitude'].values).sum()
    pred_lon = (weights * coords['longitude'].values).sum()
    b_rad = np.radians(coords['bearing'].values)
    sin_m = (weights * np.sin(b_rad)).sum()
    cos_m = (weights * np.cos(b_rad)).sum()
    pred_brg = float(np.degrees(np.arctan2(sin_m, cos_m)) % 360)
    confidence = float(probs.max()) * 100

    variants = []
    for i, (w, ci) in enumerate(zip(weights, idx)):
        row = label_to_coords.loc[ci]
        variants.append({
            'rank':      i + 1,
            'latitude':  round(float(row['latitude']), 6),
            'longitude': round(float(row['longitude']), 6),
            'bearing':   round(float(row['bearing']), 1),
            'weight':    round(float(w) * 100, 1)
        })

    return {
        'latitude':   round(float(pred_lat), 6),
        'longitude':  round(float(pred_lon), 6),
        'bearing':    round(pred_brg, 1),
        'confidence': round(confidence, 1),
        'variants':   variants
    }

# ─── Interfeys ───────────────────────────────────────────────────
st.title("📍 GPS Фото")
st.caption("[robotsoz.uz](https://robotsoz.uz)")
with st.spinner("Загрузка модели..."):
    model, label_to_coords = load_model()

uploaded = st.file_uploader(
    "Загрузите фотографию",
    type=['jpg', 'jpeg', 'png', 'webp'],
    label_visibility="collapsed"
)

if uploaded:
    img_bytes = uploaded.read()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img_bytes, caption="Загруженное фото", use_container_width=True)

    with col2:
        with st.spinner("Анализируется..."):
            result = predict(img_bytes, model, label_to_coords)

        # Ishonch darajasi rangi
        conf = result['confidence']
        if conf >= 70:
            color = "🟢"
        elif conf >= 40:
            color = "🟡"
        else:
            color = "🔴"

        st.metric("Широта",    result['latitude'])
        st.metric("Долгота",   result['longitude'])
        st.metric("Направление", f"{result['bearing']}°")
        st.metric(f"{color} Уверенность", f"{conf}%")

        maps_url = f"https://maps.google.com/?q={result['latitude']},{result['longitude']}"
        st.link_button("🗺️ Открыть в Google Maps", maps_url, use_container_width=True)

    # Top-3 variantlar
    st.divider()
    st.subheader("Топ-3 варианта")

    for v in result['variants']:
        medal = ["🥇", "🥈", "🥉"][v['rank'] - 1]
        with st.expander(f"{medal} Вариант {v['rank']} — вероятность {v['weight']}%"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Широта",     v['latitude'])
            c2.metric("Долгота",    v['longitude'])
            c3.metric("Направление", f"{v['bearing']}°")
            url = f"https://maps.google.com/?q={v['latitude']},{v['longitude']}"
            st.link_button("📍 Открыть на карте", url, use_container_width=True)