# AuraLens

An ML-powered tool that listens to a piece of classical music and identifies
the composer — not by recognizing the song, but by analyzing over 160 musical
features: harmony, timbre, rhythm, voice independence, phrase structure, and more.

---

## Composers supported

| Composer | Era |
|---|---|
| Johann Sebastian Bach | Baroque |
| Antonio Vivaldi | Baroque |
| Niccolò Paganini | Romantic |
| Pyotr Ilyich Tchaikovsky | Romantic |

---

## How it works

1. Audio is split into 30-second chunks
2. 164 features are extracted per chunk — acoustic (MFCC, spectral, rhythm, dynamics)
   and stylistic (voice independence, imitation density, register behavior,
   harmonic rhythm, phrase structure, and more)
3. A trained GradientBoosting classifier predicts the composer per chunk
4. Probabilities are averaged across all chunks for the final verdict

The model was tuned with RandomizedSearchCV using piece-level GroupKFold
cross-validation to prevent data leakage.

**Test set performance (Phase 3):**
| Composer | Chunk accuracy | Piece accuracy |
|---|---|---|
| Bach | 86.5% | 100% |
| Vivaldi | 87.3% | 100% |
| Paganini | 100% | 100% |
| Tchaikovsky | 66.8% | 92.3% |
| **Overall** | **82.5%** | **95.5%** |

> Note: the model is a closed-set classifier — it always outputs one of the
> 4 composers above. Pieces by other composers will be matched to the
> closest style it knows.

---

## Project structure
auralens/
├── data/
│ ├── features.csv # extracted features (not committed)
│ ├── features2.csv # supplemental features (not committed)
│ ├── raw/ # training audio (not committed)
│ └── processed/ # pre-chunked numpy arrays (not committed)
├── models/
│ ├── composer_model.pkl # trained composer classifier ✓ committed
│ └── era_model.pkl # trained era classifier ✓ committed
├── src/
│ ├── extract_features.py # 161-feature audio extraction
│ ├── extract_features2.py # 3 supplemental features
│ ├── train.py # model training + diagnostics
│ └── predict.py # interactive inference
├── requirements.txt
└── README.md

---
## Setup
### 1. Clone the repository
```bash
git clone https://github.com/your-username/auralens.git
cd auralens

### 2. Create a virtual environment
**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate

**macOS / Linux / WSL:**
```bash
python3 -m venv venv
source venv/bin/activate

### 3. Install Python dependencies
```bash
pip install -r requirements.txt

### 4. Install system dependency for microphone support
**Linux / WSL:**
```bash
sudo apt-get install libportaudio2

**macOS:**
```bash
brew install portaudio

**Windows:** PortAudio is bundled automatically — no extra step needed.
> Microphone recording is not supported in WSL due to hardware limitations.
> Use a file instead, or run on native Windows/Linux/macOS for microphone input.
---
## Usage
```bash
python src/predict.py

You will be prompted to choose input:
[1] Audio file (mp3, wav, flac, ogg, m4a)
[2] Record from microphone

**Option 1 — File:**
- Drag and drop your audio file onto the terminal window and press Enter
- Or paste the path (Windows, macOS, and Linux paths all accepted)
- Supported formats: mp3, wav, flac, ogg, m4a
**Option 2 — Microphone:**
- Records up to 30 seconds
- Press Ctrl+C to stop early
- Play music near your microphone while recording
---
## Retraining the model
If you have your own audio data organized as:
data/raw/
├── bach/
├── vivaldi/
├── paganini/
└── tchaikovsky/

Run in order:
```bash
python src/extract_features.py # extract 161 features → data/features.csv
python src/extract_features2.py # extract 3 more features → data/features2.csv
python src/train.py # train + evaluate → models/*.pkl

---
## Limitations
- **Closed-set classifier:** always outputs one of the 4 trained composers.
  Pieces by other composers will be matched to the stylistically closest one.
- **Chunk-level vs piece-level:** individual 30-second chunks can be uncertain,
  especially for Tchaikovsky. The piece-level result (average across all chunks)
  is the more reliable metric.
- **Instrumentation:** the model was trained primarily on orchestral and solo
  instrumental recordings. Unusual combinations (e.g., solo unaccompanied
  Romantic violin) may confuse the era classifier.
- **Microphone quality:** live recording accuracy depends on microphone quality,
  room noise, and how much of the piece is captured.