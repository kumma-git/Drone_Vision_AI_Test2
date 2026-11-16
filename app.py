
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import base64

# ==========================================
# DRONE VISION AI
# ==========================================

st.set_page_config(
    page_title="Drone Vision AI",
    page_icon="🛰️",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
CAMO_PATH = BASE_DIR / "flecktarn.jpg"
BANNER_PATH = BASE_DIR / "drone_banner.png"

IMAGE_SIZE = (224, 224)


# ==========================================
# FLECKTARN HINTERGRUND
# ==========================================

if CAMO_PATH.exists():
    image_data = base64.b64encode(
        CAMO_PATH.read_bytes()
    ).decode("utf-8")

    background_css = f"""
        background-image:
            linear-gradient(
                rgba(12, 17, 11, 0.78),
                rgba(12, 17, 11, 0.78)
            ),
            url("data:image/jpeg;base64,{image_data}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    """
else:
    background_css = "background-color: #252c22;"


# ==========================================
# DESIGN
# ==========================================

st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --olive: #89965d;
    --olive-bright: #bdc98b;
    --panel: rgba(25, 31, 23, 0.96);
    --text: #eef0e7;
    --muted: #a8ad9e;
    --line: #46503b;
}}

.stApp {{
    {background_css}
    color: var(--text);
    font-family: 'Inter', sans-serif;
}}

.block-container {{
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}

h1, h2, h3 {{
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.7px;
}}

/* Banner */

div[data-testid="stImage"] img {{
    border-radius: 14px;
    border: 1px solid #65734d;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}}

/* Panels */

.panel {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 21px;
    margin-bottom: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,.25);
}}

.panel-title {{
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #f0f2e8;
}}

.metric-label {{
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1.6px;
    font-weight: 700;
}}

.prediction {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 36px;
    color: var(--olive-bright);
    margin-top: 4px;
}}

.small-note {{
    font-size: 12px;
    color: var(--muted);
}}

/* Upload */

div[data-testid="stFileUploader"] {{
    background: #252c22;
    border: 1px dashed #89965d;
    border-radius: 13px;
    padding: 14px;
}}

/* Buttons */

.stButton > button {{
    background: #89965d;
    color: #11150f;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
}}

.stButton > button:hover {{
    background: #b1bd82;
    color: #10140e;
}}

.stProgress > div > div > div > div {{
    background-color: #a3ad73;
}}

hr {{
    border-color: #41483a;
}}

footer {{
    visibility: hidden;
}}

</style>
""", unsafe_allow_html=True)


# ==========================================
# BANNER GANZ OBEN
# ==========================================

if BANNER_PATH.exists():
    st.image(
        str(BANNER_PATH),
        use_container_width=True
    )
else:
    st.warning(
        "Banner nicht gefunden. "
        "Bitte drone_banner.png in dein GitHub-Repository hochladen."
    )


# ==========================================
# MODELL LADEN
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ==========================================
# LABELS LADEN
# ==========================================

@st.cache_data
def load_labels():

    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)

    return labels


# ==========================================
# BILDERKENNUNG
# ==========================================

def predict_image(image, model, labels):

    image = image.convert("RGB").resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)
    array = (array / 127.5) - 1.0
    array = np.expand_dims(array, axis=0)

    scores = model.predict(array, verbose=0)[0]

    if len(scores) != len(labels):
        raise ValueError(
            f"Das Modell liefert {len(scores)} Werte, "
            f"aber labels.txt enthält {len(labels)} Labels."
        )

    return scores


# ==========================================
# HAUPTBEREICH
# ==========================================

left, right = st.columns(
    [1.05, 1.35],
    gap="large"
)


# ==========================================
# LINKE SEITE
# ==========================================

with left:

    st.html("""
    <div class="panel">
        <div class="panel-title">01 / BILDEINGABE</div>
    </div>
    """)

    st.write(
        "Lade ein Bild hoch, das von deinem "
        "trainierten KI-Modell analysiert werden soll."
    )

    uploaded_file = st.file_uploader(
        "Bild auswählen",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.html("""
    <p class="small-note">
        Unterstützte Formate: JPG, JPEG, PNG, WEBP
    </p>
    """)

    st.html("""
    <div class="panel">
        <div class="panel-title">02 / MODELLSTATUS</div>

        <p>
            <span style="color:#b9c78a;">●</span>
            Modell-Datei: <b>keras_model.h5</b>
        </p>

        <p>
            <span style="color:#b9c78a;">●</span>
            Klassen-Datei: <b>labels.txt</b>
        </p>

        <p>
            <span style="color:#b9c78a;">●</span>
            Hintergrund: <b>Flecktarn</b>
        </p>

        <p class="small-note">
            Eingabegröße: 224 × 224 Pixel
        </p>
    </div>
    """)


# ==========================================
# RECHTE SEITE
# ==========================================

with right:

    st.html("""
    <div class="panel">
        <div class="panel-title">03 / ANALYSEERGEBNIS</div>
    </div>
    """)

    if uploaded_file is None:

        st.info(
            "Lade links ein Bild hoch, "
            "um die Klassifizierung zu starten."
        )

    else:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Hochgeladenes Bild",
            use_container_width=True
        )

        try:

            model = load_model()
            labels = load_labels()

            scores = predict_image(
                image,
                model,
                labels
            )

            best_index = int(np.argmax(scores))
            best_label = labels[best_index]
            confidence = float(scores[best_index]) * 100

            st.html("""
            <div class="metric-label">
                WAHRSCHEINLICHSTE KLASSE
            </div>
            """)

            st.html(f"""
            <div class="prediction">{best_label}</div>

            <p class="small-note">
                Modell-Konfidenz: <b>{confidence:.1f}%</b>
            </p>
            """)

            st.divider()

            st.markdown("**Wahrscheinlichkeiten aller Klassen**")

            order = np.argsort(scores)[::-1]

            for index in order:

                percentage = float(scores[index]) * 100

                st.write(
                    f"{labels[index]} — {percentage:.1f}%"
                )

                st.progress(
                    float(np.clip(scores[index], 0, 1))
                )

        except Exception as error:

            st.error(
                f"Das Modell konnte nicht ausgeführt werden: {error}"
            )


# ==========================================
# FOOTER
# ==========================================

st.html("""
<div style="
    text-align:center;
    color:#a0a68f;
    font-size:11px;
    margin-top:25px;
    letter-spacing:1px;
">
    DRONE VISION AI
    <br>
    AI IMAGE CLASSIFICATION SYSTEM
    <br><br>
    EDUCATIONAL PROJECT · MODEL PREDICTIONS MAY BE INCORRECT
</div>
""")
