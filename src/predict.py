#!/usr/bin/env python3
"""
predict.py — upload an audio file and identify the composer
"""

import os
import sys
import numpy as np
import librosa
import joblib
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# import feature extraction from the same src/ folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import (
    extract_mfcc_features,
    extract_spectral_features,
    extract_harmony_features,
    extract_rhythm_features,
    extract_dynamics_features,
    extract_texture_features,
    extract_structure_features,
    extract_polyphony_features,
    SR, CHUNK_DURATION, N_FFT, HOP_LENGTH,
)

# ─── paths ───────────────────────────────────────────────────
BASE_DIR            = os.path.expanduser("~/auralens")
MODELS_DIR          = os.path.join(BASE_DIR, "models")
COMPOSER_MODEL_PATH = os.path.join(MODELS_DIR, "composer_model.pkl")
ERA_MODEL_PATH      = os.path.join(MODELS_DIR, "era_model.pkl")

# ─── display names ───────────────────────────────────────────
COMPOSER_FULL = {
    'bach':        'Johann Sebastian Bach',
    'vivaldi':     'Antonio Vivaldi',
    'paganini':    'Niccolò Paganini',
    'tchaikovsky': 'Pyotr Ilyich Tchaikovsky',
}
COMPOSER_LAST = {
    'bach':        'Bach',
    'vivaldi':     'Vivaldi',
    'paganini':    'Paganini',
    'tchaikovsky': 'Tchaikovsky',
}
ERA_DISPLAY = {
    'baroque':  'Baroque',
    'romantic': 'Romantic',
}

# must match what was excluded during training
EXCLUDED_FEATURES = {'key_mode'}

# musical labels for top features
FEATURE_LABELS = {
    'spectral_bandwidth':        ('Timbral richness',      'width of the harmonic spread'),
    'mfcc_mean_0':               ('Overall energy',        'total spectral energy'),
    'mfcc_mean_1':               ('Spectral slope',        'balance of bass vs treble'),
    'mfcc_mean_2':               ('Tonal colour',          'brightness vs warmth of the sound'),
    'mfcc_mean_3':               ('Spectral shape',        'curvature of the frequency envelope'),
    'mfcc_mean_6':               ('Mid-range texture',     'character of mid-frequency content'),
    'mfcc_mean_7':               ('Timbral texture',       'fine-grained spectral texture'),
    'mfcc_mean_10':              ('Upper harmonics',       'character of upper harmonic content'),
    'mfcc_mean_12':              ('Finest texture',        'highest-order timbral detail'),
    'mfcc_std_1':                ('Spectral variation',    'how much the spectral slope varies'),
    'mfcc_std_11':               ('Timbral variation',     'how much the timbre changes over time'),
    'mfcc_std_12':               ('Texture variation',     'variability in fine timbral detail'),
    'rms_mean':                  ('Loudness',              'average loudness of the piece'),
    'rms_std':                   ('Dynamic variation',     'how much the loudness fluctuates'),
    'dynamic_range':             ('Dynamic range',         'difference between loudest and quietest'),
    'loudness_slope':            ('Loudness trend',        'whether the piece grows louder or quieter'),
    'spectral_centroid':         ('Brightness',            'centre of mass of the spectrum'),
    'spectral_rolloff':          ('Frequency ceiling',     'frequency below which 85% of energy falls'),
    'spectral_flatness':         ('Tone purity',           'how pure and tonal the sound is'),
    'spectral_flux':             ('Spectral activity',     'how fast the spectrum changes'),
    'spectral_contrast_0':       ('Bass contrast',         'peak-valley difference in bass band'),
    'spectral_contrast_1':       ('Low contrast',          'contrast in lower frequencies'),
    'spectral_contrast_2':       ('Mid contrast',          'contrast in mid frequencies'),
    'spectral_contrast_5':       ('Treble contrast',       'contrast in treble frequencies'),
    'spectral_contrast_6':       ('Brilliance contrast',   'sharpness of the highest peaks'),
    'spectral_peak_count':       ('Polyphony density',     'number of simultaneous frequency peaks'),
    'polyphony_estimate':        ('Voice count',           'estimated simultaneous voices'),
    'hp_ratio':                  ('Harmonic purity',       'ratio of melodic to percussive content'),
    'zcr_mean':                  ('Noisiness',             'how noisy vs tonal the texture is'),
    'tempo':                     ('Tempo',                 'estimated beats per minute'),
    'onset_rate':                ('Note density',          'how many notes per second'),
    'rhythm_regularity':         ('Rhythmic regularity',   'consistency of note timing'),
    'tempogram_entropy':         ('Rhythmic complexity',   'spread of rhythmic energy'),
    'syncopation':               ('Syncopation',           'fraction of notes falling off the beat'),
    'tonal_instability':         ('Harmonic movement',     'how rapidly the harmony shifts'),
    'chroma_entropy':            ('Key focus',             'how concentrated the pitch classes are'),
    'key_clarity':               ('Key strength',          'how strongly one key dominates'),
    'chromaticism':              ('Chromaticism',          'use of notes outside the home key'),
    'autocorr_peak':             ('Repetition',            'how strongly the piece repeats itself'),
    'self_similarity_peakiness': ('Structural contrast',   'variety between sections'),
}
# ─────────────────────────────────────────────────────────────


def make_bar(value, width=12):
    """Filled progress bar. value must be 0.0 – 1.0."""
    n = max(0, min(width, int(round(value * width))))
    return '█' * n + '░' * (width - n)


def extract_chunk_features(y, sr, S):
    """Extract all features from one 30-second chunk of audio."""
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr, n_fft=N_FFT)
    row = {}
    row.update(extract_mfcc_features(y, sr))
    row.update(extract_spectral_features(S, sr))
    row.update(extract_harmony_features(sr, chroma))
    row.update(extract_rhythm_features(y, sr))
    row.update(extract_dynamics_features(y, sr, S))
    row.update(extract_texture_features(y))
    row.update(extract_structure_features(chroma))
    row.update(extract_polyphony_features(S))
    return row


def load_and_chunk(file_path):
    """Load audio file and extract features from each 30-second window."""
    duration = librosa.get_duration(path=file_path)
    print(f"  duration : {duration:.1f}s  →  ", end='', flush=True)

    chunks = []
    start  = 0.0

    while start + CHUNK_DURATION <= duration:
        y, sr = librosa.load(file_path, sr=SR,
                              offset=start, duration=CHUNK_DURATION)
        if len(y) >= SR * 5:
            S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
            chunks.append(extract_chunk_features(y, sr, S))
        start += CHUNK_DURATION

    # handle audio shorter than 30 seconds (min 10 s)
    if not chunks:
        if duration >= 10:
            y, sr = librosa.load(file_path, sr=SR)
            S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
            chunks.append(extract_chunk_features(y, sr, S))
        else:
            raise ValueError(
                f"Audio too short ({duration:.1f}s) — need at least 10 seconds")

    print(f"{len(chunks)} chunk(s) analysed")
    return chunks


def chunks_to_matrix(chunks):
    """Convert list of feature dicts to a numpy matrix.
    Excludes features that were dropped during training."""
    feature_names = [k for k in chunks[0].keys() if k not in EXCLUDED_FEATURES]
    X = np.array([[chunk[k] for k in feature_names] for chunk in chunks])
    return X, feature_names


def display_results(composer_probs, composer_classes,
                    era_probs, era_classes):
    """Print era + composer probability bars and verdict."""
    best_composer = composer_classes[np.argmax(composer_probs)]
    best_era      = era_classes[np.argmax(era_probs)]

    print()
    print("─────────────────────────────────────────")
    print("  ERA ANALYSIS")
    for cls, prob in sorted(zip(era_classes, era_probs), key=lambda x: -x[1]):
        name = ERA_DISPLAY.get(cls, cls.capitalize())
        print(f"  {name:<12} {make_bar(prob)}  {prob * 100:.0f}%")

    print()
    print("  COMPOSER ANALYSIS")
    for cls, prob in sorted(zip(composer_classes, composer_probs),
                             key=lambda x: -x[1]):
        name = COMPOSER_LAST.get(cls, cls.capitalize())
        print(f"  {name:<12} {make_bar(prob)}  {prob * 100:.0f}%")

    print()
    print(f"  VERDICT: {COMPOSER_FULL.get(best_composer, best_composer)}"
          f" — {ERA_DISPLAY.get(best_era, best_era)} period")
    print("─────────────────────────────────────────")


def display_feature_insights(composer_model, feature_names, X, top_n=5):
    """Show the top features that drove the prediction in musical terms."""
    clf    = (composer_model.named_steps.get('gb')
               or composer_model.named_steps.get('rf'))
    scaler = composer_model.named_steps['scaler']

    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_n]

    # z-score each feature relative to the training distribution
    X_scaled = scaler.transform(X)
    z_mean   = np.mean(X_scaled, axis=0)   # average across chunks

    print("\n  KEY CHARACTERISTICS")
    print("─────────────────────────────────────────")

    for idx in top_indices:
        fname = feature_names[idx]
        z     = z_mean[idx]

        label, desc = FEATURE_LABELS.get(fname, (fname, ''))

        # bar: map z-score [-3, +3] → [0, 1]
        b = make_bar((z + 3) / 6.0, width=16)

        if   z >  1.5: level = 'notably high'
        elif z >  0.5: level = 'above average'
        elif z > -0.5: level = 'average'
        elif z > -1.5: level = 'below average'
        else:          level = 'notably low'

        print(f"  {label}")
        print(f"  {b}  {level}")
        if desc:
            print(f"  ↳ {desc}")
        print()


def main():
    print("═════════════════════════════════════════")
    print("  AURALENS — composer identifier")
    print("═════════════════════════════════════════")

    # ── check models exist ────────────────────────────────────
    for path, name in [(COMPOSER_MODEL_PATH, 'composer'),
                       (ERA_MODEL_PATH, 'era')]:
        if not os.path.exists(path):
            print(f"\nERROR: {name} model not found at {path}")
            print("       run train.py first")
            sys.exit(1)

    composer_model = joblib.load(COMPOSER_MODEL_PATH)
    era_model      = joblib.load(ERA_MODEL_PATH)
    print("  models loaded ✓")

    # ── get file path from user ───────────────────────────────
    print()
    print("  Paste the path to your audio file and press Enter.")
    print("  Supported: mp3, wav, flac, ogg, m4a")
    print()

    while True:
        raw = input("  > ").strip().strip('"').strip("'")
        if not raw:
            continue
        file_path = os.path.expanduser(raw)
        if os.path.isfile(file_path):
            break
        print(f"  File not found: {file_path}")
        print("  Try again.")
        print()

    # ── run pipeline ──────────────────────────────────────────
    try:
        print("\n[ extracting features ]")
        chunks = load_and_chunk(file_path)
        X, feature_names = chunks_to_matrix(chunks)

        composer_probs   = np.mean(composer_model.predict_proba(X), axis=0)
        era_probs        = np.mean(era_model.predict_proba(X), axis=0)
        composer_classes = composer_model.classes_
        era_classes      = era_model.classes_

        display_results(composer_probs, composer_classes,
                        era_probs, era_classes)
        display_feature_insights(composer_model, feature_names, X, top_n=5)

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
