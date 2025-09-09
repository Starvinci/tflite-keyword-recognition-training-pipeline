## LOME Keyword-Spotting Pipeline

Diese Pipeline wurde aufgebaut, um ein Audio‑Klassifizierungsmodell auf dem Coral Edge TPU zu betreiben.

Hinweise zur Datensammlung:
- Mindestens 70 Aufnahmen a 1 Sekunde pro Aufnahme.
- Vielfältige Hintergründe (z. B. Stille, Konversationen, Haushaltsgeräusche).
- In der Praxis waren mehrere Iterationen aus Datensammeln und Training nötig, bis das Modell zuverlässig nutzbar war.

Diese Repo stellt eine eigenständige Trainings-Pipeline für Keyword-Spotting bereit.

### Struktur
- `data/keyword`: Rohaufnahmen der Zielwörter (Unterordner je Wort)
- `data/hintergrund`: Hintergrundgeräusche als WAV
- `export/curated`: Kuratierte Daten
- `export/mixed`: Gemischte Daten (Zielwörter + Hintergrund, inkl. `_other`, `_background`)
- `export/features`: Extrahierte STFT-Features (`.npz`)
- `export/models`: Modelle (`.keras`, `.tflite`, optional `.h`)
- `src`: Pipeline-Module
- `scripts`: Hilfsskripte (Aufnahme, Snippet-Cutter)
- `tests`: einfache Inferenztests

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Daten vorbereiten
- In `data/keyword` gilt: JEDES Keyword hat einen eigenen Unterordner (z. B. `data/keyword/lome/`, `data/keyword/notfall/`) und darin liegen die Audio-Dateien für dieses Keyword.
- In `data/hintergrund` können die Hintergrund-Audios direkt in den Ordner abgelegt werden (kein Unterordner nötig).

Optional: Längere WAVs in 1s-Snippets schneiden:
```bash
python scripts/cut_snippets.py --in_file pfad/zur/datei.wav --out_dir data/keyword/lome
```

Audio aufnehmen:
```bash
python scripts/record_audio.py --out_file data/keyword/lome/sample.wav --duration 3 --sample_rate 16000
```

### Pipeline ausführen
```bash
python start_trainings_pipeline.py \
  --targets "<keyword_1>,<keyword_2>" \
  --max_per_class 0 \
  --sample_rate 16000 \
  --sample_time 1.0 \
  --bg_snippets_per_file 5 \
  --epochs 100 \
  --trials 5 \
  --export_header
```

Hinweise zur Verwendung:
- Ersetze `<keyword_1>,<keyword_2>` bei `--targets` durch deine Zielwörter. Beispiel: `--targets "lome,notfall"`.
- Lege Audios zu jedem Keyword in `data/keyword/<keyword>/` ab, z. B. `data/keyword/lome/`.
- Lege Hintergrund-Audios direkt in `data/hintergrund/`.
- Für einen schnellen Test kannst du mit kleinen Werten starten, z. B. `--epochs 1 --trials 1`.

Ergebnis:
- `export/models/model.keras`
- `export/models/model.tflite`
- Optional: `export/models/model.h`

### Tests
```bash
python -m pytest -q
```

### Lizenz/Urheberrecht
Copyright (c) 2025, Starvinci. Alle Rechte vorbehalten.


