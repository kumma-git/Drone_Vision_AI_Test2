# Drone Vision AI – Streamlit

## Dateien
- `app.py` – Streamlit-Oberfläche und Bildklassifizierung
- `keras_model.h5` – dein trainiertes Keras-Modell
- `labels.txt` – Klassen in derselben Reihenfolge wie die Modellausgabe
- `requirements.txt` – Python-Abhängigkeiten
- `.python-version` – Python-Version für Streamlit Cloud

## Lokal starten
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Auf GitHub / Streamlit Community Cloud
1. Lade alle Dateien in ein GitHub-Repository hoch.
2. Lade `keras_model.h5` zusätzlich in denselben Ordner wie `app.py`.
3. Verbinde das Repository mit Streamlit Community Cloud.
4. Wähle `app.py` als Main file path.

Die App erwartet ein Bildmodell mit Eingabegröße 224×224×3 und Teachable-Machine-Normalisierung von -1 bis 1.
Die Labels müssen in `labels.txt` in exakt derselben Reihenfolge wie die Ausgabeklassen stehen.
