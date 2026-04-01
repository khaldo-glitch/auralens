# Auralens

An ML-powered tool that listens to a piece of classical music and identifies 
the composer's style — not by recognizing the song, but by analyzing musical 
features like harmony, timbre, and rhythm.

## Composers supported
- Johann Sebastian Bach
- Antonio Vivaldi
- Niccolò Paganini
- Pyotr Ilyich Tchaikovsky

## Project structure
- `src/extract_features.py` — extracts audio features from music files
- `src/train.py` — trains the classifier model
- `src/predict.py` — runs prediction on new audio input
- `data/raw/` — training audio files (not committed to git)
- `models/` — saved trained model (not committed to git)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
