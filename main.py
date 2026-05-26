# app.py
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
import numpy as np
import io

app = Flask(__name__)

# ─── Model yuklash ────────────────────────────────────────────────
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

label_to_coords = data['label_to_coords']

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

def predict(image_bytes, top_k=3):
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
    confidence = float(probs.max())

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
        'confidence': round(confidence * 100, 1),
        'variants':   variants
    }

# ─── HTML ─────────────────────────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Определение местоположения</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: sans-serif; background: #f0f2f5; color: #1a1a2e; }
  .container { max-width: 720px; margin: 40px auto; padding: 0 16px 60px; }
  h1 { font-size: 22px; font-weight: 600; margin-bottom: 24px; }

  .upload-box {
    border: 2px dashed #4a90d9; border-radius: 12px;
    padding: 40px; text-align: center; background: #fff;
    cursor: pointer; transition: background 0.2s;
  }
  .upload-box:hover { background: #f0f7ff; }
  .upload-box input { display: none; }
  .upload-icon { font-size: 40px; margin-bottom: 12px; }
  .upload-box p { color: #666; font-size: 15px; }
  .upload-box span { color: #4a90d9; font-weight: 500; }

  #preview-wrap { margin-top: 20px; display: none; }
  #preview { width: 100%; border-radius: 10px; max-height: 340px; object-fit: cover; }

  .btn {
    display: block; width: 100%; margin-top: 16px;
    padding: 14px; background: #4a90d9; color: #fff;
    border: none; border-radius: 10px; font-size: 16px;
    cursor: pointer; transition: background 0.2s;
  }
  .btn:hover { background: #357abd; }
  .btn:disabled { background: #aaa; cursor: not-allowed; }

  #result { margin-top: 28px; display: none; }

  .card {
    background: #fff; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .card h2 { font-size: 13px; color: #999; font-weight: 500;
             letter-spacing: 0.05em; margin-bottom: 16px; }

  .coord-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .coord-item {
    flex: 1; min-width: 130px;
    background: #f7f9fc; border-radius: 8px; padding: 12px 16px;
  }
  .coord-item .label { font-size: 12px; color: #999; margin-bottom: 4px; }
  .coord-item .value { font-size: 18px; font-weight: 600; color: #1a1a2e; }

  .conf-wrap { margin-top: 16px; }
  .conf-label { font-size: 13px; color: #666; margin-bottom: 6px; }
  .conf-bar { height: 8px; background: #e8edf2; border-radius: 4px; overflow: hidden; }
  .conf-fill { height: 100%; border-radius: 4px; background: #4a90d9; transition: width 0.6s; }

  .maps-btn {
    display: inline-block; margin-top: 16px;
    padding: 10px 20px; background: #34a853;
    color: #fff; border-radius: 8px; text-decoration: none;
    font-size: 14px; transition: background 0.2s;
  }
  .maps-btn:hover { background: #2d9248; }

  .variant-row {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid #f0f0f0; font-size: 14px;
  }
  .variant-row:last-child { border-bottom: none; }
  .rank {
    width: 26px; height: 26px; border-radius: 50%;
    background: #e8edf2; display: flex; align-items: center;
    justify-content: center; font-weight: 600; font-size: 13px; flex-shrink: 0;
  }
  .rank.r1 { background: #4a90d9; color: #fff; }
  .vcoord { flex: 1; color: #333; }
  .vw { color: #4a90d9; font-weight: 500; font-size: 13px; }

  .spinner { display: none; text-align: center; padding: 30px; color: #666; }
  .spinner.show { display: block; }

  .arrow { display: inline-block; font-size: 22px; transition: transform 0.5s; }
</style>
</head>
<body>
<div class="container">
  <h1>📍 Определение местоположения по фото</h1>

  <div class="upload-box" id="drop-zone"
       onclick="document.getElementById('file-input').click()">
    <input type="file" id="file-input" accept="image/*" onchange="onFile(this)">
    <div class="upload-icon">🖼️</div>
    <p><span>Нажмите для выбора фото</span> или перетащите сюда</p>
    <p style="margin-top:6px;font-size:13px;color:#aaa">JPG, PNG, WEBP</p>
  </div>

  <div id="preview-wrap">
    <img id="preview" src="" alt="preview">
    <button class="btn" id="predict-btn" onclick="predict()">
      🔍 Определить местоположение
    </button>
  </div>

  <div class="spinner" id="spinner">⏳ Анализируется...</div>

  <div id="result">
    <div class="card">
      <h2>РЕЗУЛЬТАТ</h2>
      <div class="coord-row">
        <div class="coord-item">
          <div class="label">Широта</div>
          <div class="value" id="r-lat">—</div>
        </div>
        <div class="coord-item">
          <div class="label">Долгота</div>
          <div class="value" id="r-lon">—</div>
        </div>
        <div class="coord-item">
          <div class="label">Направление</div>
          <div class="value">
            <span class="arrow" id="r-arrow">↑</span>
            <span id="r-brg">—</span>°
          </div>
        </div>
      </div>
      <div class="conf-wrap">
        <div class="conf-label">Уверенность: <b id="r-conf">—</b>%</div>
        <div class="conf-bar">
          <div class="conf-fill" id="r-conf-bar" style="width:0%"></div>
        </div>
      </div>
      <a id="maps-link" href="#" target="_blank" class="maps-btn">
        🗺️ Открыть в Google Maps
      </a>
    </div>

    <div class="card">
      <h2>ТОП-3 ВАРИАНТА</h2>
      <div id="variants"></div>
    </div>
  </div>
</div>

<script>
let selectedFile = null;

function onFile(input) {
  if (!input.files[0]) return;
  selectedFile = input.files[0];
  document.getElementById('preview').src = URL.createObjectURL(selectedFile);
  document.getElementById('preview-wrap').style.display = 'block';
  document.getElementById('result').style.display = 'none';
}

const zone = document.getElementById('drop-zone');
zone.addEventListener('dragover',  e => { e.preventDefault(); zone.style.background = '#f0f7ff'; });
zone.addEventListener('dragleave', () => { zone.style.background = ''; });
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.style.background = '';
  const f = e.dataTransfer.files[0];
  if (!f) return;
  selectedFile = f;
  document.getElementById('preview').src = URL.createObjectURL(f);
  document.getElementById('preview-wrap').style.display = 'block';
  document.getElementById('result').style.display = 'none';
});

async function predict() {
  if (!selectedFile) return;
  document.getElementById('predict-btn').disabled = true;
  document.getElementById('spinner').classList.add('show');
  document.getElementById('result').style.display = 'none';

  const fd = new FormData();
  fd.append('image', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const d = await res.json();

    document.getElementById('r-lat').textContent  = d.latitude;
    document.getElementById('r-lon').textContent  = d.longitude;
    document.getElementById('r-brg').textContent  = d.bearing;
    document.getElementById('r-conf').textContent = d.confidence;
    document.getElementById('r-conf-bar').style.width = d.confidence + '%';
    document.getElementById('r-arrow').style.transform = `rotate(${d.bearing}deg)`;
    document.getElementById('maps-link').href =
      `https://maps.google.com/?q=${d.latitude},${d.longitude}`;

    document.getElementById('variants').innerHTML = d.variants.map(v => `
      <div class="variant-row">
        <div class="rank ${v.rank===1?'r1':''}">${v.rank}</div>
        <div class="vcoord">${v.latitude}, ${v.longitude} · ${v.bearing}°</div>
        <div class="vw">${v.weight}%</div>
      </div>`).join('');

    document.getElementById('result').style.display = 'block';
  } catch(e) {
    alert('Ошибка: ' + e.message);
  } finally {
    document.getElementById('predict-btn').disabled = false;
    document.getElementById('spinner').classList.remove('show');
  }
}
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    result = predict(request.files['image'].read())
    return jsonify(result)

if __name__ == '__main__':
    print("✅ Сервер запущен: http://localhost:5000")
    app.run(debug=False, port=5000)