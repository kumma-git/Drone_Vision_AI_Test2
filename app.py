import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -------------------- Einstellungen --------------------
st.set_page_config(
    page_title="Drone Vision AI",
    page_icon="🛰️",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
IMAGE_SIZE = (224, 224)

# -------------------- Design --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --olive: #a3a875;
    --olive-bright: #c4cb8a;
    --panel: #20251e;
    --panel-light: #292f26;
    --text: #eef0e7;
    --muted: #a8ad9e;
    --line: #41483a;
}
.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(119, 132, 76, .16), transparent 38%),
        #141812;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
h1, h2, h3 {font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: .7px;}
.hero {
    background: linear-gradient(115deg, #303a2a, #20271e 65%, #252b21);
    border: 1px solid #4b563d;
    border-radius: 18px;
    padding: 27px 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 35px rgba(0,0,0,.22);
}
.eyebrow {color: var(--olive-bright); font-size: 11px; letter-spacing: 2.4px; font-weight: 700;}
.hero h1 {font-size: 42px; margin: 5px 0 0; color: #f3f4ec;}
.hero p {color: #c0c6b5; margin: 5px 0 0; font-size: 14px;}
.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 21px;
    margin-bottom: 16px;
}
.panel-title {font-size: 21px; font-weight: 700; margin-bottom: 12px; color: #f0f2e8;}
.metric-label {font-size: 11px; color: var(--muted); letter-spacing: 1.6px; font-weight: 700;}
.prediction {font-family: 'Barlow Condensed', sans-serif; font-size: 34px; color: var(--olive-bright); margin-top: 4px;}
.small-note {font-size: 12px; color: var(--muted);}
div[data-testid="stFileUploader"] {
    background: #252c22; border: 1px dashed #77815e;
    border-radius: 13px; padding: 14px;
}
.stButton > button {
    background: #7e8958; color: #11150f; border: 0;
    border-radius: 9px; font-weight: 700;
}
.stButton > button:hover {background: #a3ad73; color: #10140e;}
hr {border-color: #41483a;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------- Modell und Labels --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_labels():
    labels = []
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # Unterstützt z. B. "0 Flughafen" und "Flughafen"
            parts = line.split(maxsplit=1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return labels

def predict_image(image, model, labels):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = (array / 127.5) - 1.0  # Teachable-Machine-Bildmodelle
    array = np.expand_dims(array, axis=0)
    scores = model.predict(array, verbose=0)[0]
    if len(scores) != len(labels):
        raise ValueError(
            f"Das Modell liefert {len(scores)} Werte, aber labels.txt enthält "
            f"{len(labels)} Labels."
        )
    return scores

# -------------------- Kopfbereich --------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">IMAGE CLASSIFICATION SYSTEM · DEMO</div>
  <h1>DRONE VISION AI</h1>
  <p>KI-gestützte Bildklassifizierung · Lokaler Modellbetrieb innerhalb dieser Streamlit-App</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.markdown('<div class="panel"><div class="panel-title">01 / BILDEINGABE</div>', unsafe_allow_html=True)
    st.write("Lade ein Bild hoch, das von deinem trainierten Modell klassifiziert werden soll.")
    uploaded_file = st.file_uploader(
        "Bild auswählen",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.markdown('<p class="small-note">Unterstützte Formate: JPG, JPEG, PNG, WEBP</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">02 / MODELLSTATUS</div>', unsafe_allow_html=True)
    st.markdown('<span style="color:#b9c78a">●</span> Modell-Datei: <b>keras_model.h5</b>', unsafe_allow_html=True)
    st.markdown('<span style="color:#b9c78a">●</span> Klassen-Datei: <b>labels.txt</b>', unsafe_allow_html=True)
    st.markdown('<p class="small-note">Eingabegröße: 224 × 224 Pixel</p></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="panel-title">03 / ANALYSEERGEBNIS</div>', unsafe_allow_html=True)
    if uploaded_file is None:
        st.info("Lade links ein Bild hoch, um die Klassifizierung zu starten.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)
        try:
            model = load_model()
            labels = load_labels()
            scores = predict_image(image, model, labels)
            best_index = int(np.argmax(scores))
            best_label = labels[best_index]
            confidence = float(scores[best_index]) * 100

            st.markdown('<div class="metric-label">WAHRSCHEINLICHSTE KLASSE</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction">{best_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="small-note">Modell-Konfidenz: <b>{confidence:.1f}%</b></p>', unsafe_allow_html=True)
            st.divider()
            st.markdown("**Wahrscheinlichkeiten aller Klassen**")
            order = np.argsort(scores)[::-1]
            for index in order:
                st.write(f"{labels[index]} — {float(scores[index]) * 100:.1f}%")
                st.progress(float(np.clip(scores[index], 0, 1)))
        except Exception as error:
            st.error(f"Das Modell konnte nicht ausgeführt werden: {error}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#858d7a; font-size:11px; margin-top:22px;">
  DRONE VISION AI · EDUCATIONAL IMAGE-CLASSIFICATION DEMO
  <br>Modellvorhersagen können falsch sein und sind keine verlässliche Grundlage für Entscheidungen.
</div>
""", unsafe_allow_html=True)
