# AuraLens

An ML-powered tool that listens to classical music and identifies the **composer** — not by recognizing the piece, but by analyzing over **160 musical features** including harmony, timbre, rhythm, voice independence, and phrase structure.

---

## Composers supported

| Composer                 | Era      |
| ------------------------ | -------- |
| Johann Sebastian Bach    | Baroque  |
| Antonio Vivaldi          | Baroque  |
| Niccolò Paganini         | Romantic |
| Pyotr Ilyich Tchaikovsky | Romantic |

---

## How it works

1. Audio is split into **30-second chunks**
2. **164 features** are extracted per chunk — acoustic (MFCC, spectral, rhythm, dynamics) and stylistic (voice independence, imitation density, register behavior, harmonic rhythm, phrase structure, etc.)
3. A **GradientBoosting classifier** predicts the composer for each chunk
4. Chunk probabilities are **averaged** to produce the final piece-level prediction

The model was tuned with **RandomizedSearchCV** using **piece-level GroupKFold** to avoid data leakage.

> **Note:** This is a *closed-set* classifier — it always outputs one of the four composers above.
> Music by other composers will be matched to the closest style it knows.

---

## Project structure

```
auralens/
├── data/
│   ├── features.csv          # extracted features (not committed)
│   ├── features2.csv         # supplemental features (not committed)
│   ├── raw/                  # training audio (not committed)
│   └── processed/            # pre-chunked numpy arrays (not committed)
├── models/
│   ├── composer_model.pkl    # trained composer classifier ✓ committed
│   └── era_model.pkl         # trained era classifier ✓ committed
├── src/
│   ├── extract_features.py   # 161-feature audio extraction
│   ├── extract_features2.py  # 3 supplemental features
│   ├── train.py              # model training + diagnostics
│   └── predict.py            # interactive inference
├── requirements.txt
└── README.md
```

---

## Setup

### 0. Install Python (if not already installed)

AuraLens requires **Python 3.9+**.

* Download Python from: https://www.python.org/downloads/
* During installation on Windows, **make sure to check**:

  ```
  ✔ Add Python to PATH
  ```

To verify installation:

```bash
python --version
```

or (on some systems):

```bash
python3 --version
```

---

### 1. Clone the repository

```bash
git clone https://github.com/your-username/auralens.git
cd auralens
```

---

### 2. Create a virtual environment

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**If activation fails (PowerShell only):**

Run the following command once, then try activating again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

Then:

```powershell
venv\Scripts\activate
```

---

**macOS / Linux / WSL:**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install system dependency for microphone support

**Linux / WSL:**

```bash
sudo apt-get install libportaudio2
```

**macOS:**

```bash
brew install portaudio
```

**Windows:**
PortAudio is bundled automatically — no extra step needed.

> Microphone recording is **not supported in WSL** due to hardware limitations.
> Use an audio file instead, or run on native Windows/Linux/macOS for microphone input.

---

## Usage

```bash
python src/predict.py
```

You will be prompted to choose input:

```
[1] Audio file (mp3, wav, flac, ogg, m4a)
[2] Record from microphone
```

### Option 1 — File

* Drag and drop your audio file into the terminal, then press Enter
* Or paste the file path manually
* Supported formats: **mp3, wav, flac, ogg, m4a**

### Option 2 — Microphone

* Records up to **30 seconds**
* Press **Ctrl+C** to stop early
* Play music near your microphone while recording

---

## Retraining the model

If you have your own dataset structured as:

```
data/raw/
├── bach/
├── vivaldi/
├── paganini/
└── tchaikovsky/
```

Run the pipeline:

```bash
python src/extract_features.py     # extract 161 features → data/features.csv
python src/extract_features2.py    # extract 3 more features → data/features2.csv
python src/train.py                # train + evaluate → models/*.pkl
```

---

## Future Improvements

AuraLens is an evolving project, and several enhancements are planned to expand its accuracy, scope, and usability:

* **Improved composer detection**
  Further tuning of the feature set and model architecture to increase chunk-level and piece-level accuracy, especially for closely related styles.

* **Deeper stylistic analysis**
  More emphasis on high-level musical traits such as phrasing behavior, harmonic language, orchestration patterns, and voice-leading tendencies unique to each composer.

* **Support for additional composers**
  Expanding the dataset to include more Baroque, Classical, and Romantic composers.

* **Addition of the Classical era**
  Introducing composers such as Mozart, Haydn, and early Beethoven to cover the stylistic gap between Baroque and Romantic periods.

* **Enhanced feature extraction pipeline**
  Incorporating more advanced spectral, rhythmic, and structural descriptors for richer musical representation.

* **Improved user interface**
  A cleaner, more intuitive CLI and (eventually) a lightweight GUI for easier interaction and real-time feedback.

* **Better analysis output**
  More detailed explanations of why the model chose a certain composer, including feature importance and stylistic reasoning.

---

## Limitations

* **Closed-set classifier:** always outputs one of the four trained composers
* **Chunk-level uncertainty:** individual 30-second chunks may be noisy, especially for Tchaikovsky
* **Instrumentation bias:** trained mostly on orchestral + solo instrumental recordings
* **Microphone quality:** live inference depends heavily on mic quality, room noise, and recording duration
