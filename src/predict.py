#!/usr/bin/env python3
"""
predict.py — AuraLens composer identifier
Supports: file path input (drag & drop or paste), microphone recording
Cross-platform: Windows, macOS, Linux
"""

import os
import sys
import time
import tempfile
import warnings
import numpy as np
import librosa
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

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
    extract_horizontal_vertical_features,
    extract_register_features,
    extract_voice_independence_features,
    extract_melodic_motion_features,
    extract_imitation_features,
    extract_motivic_features,
    extract_phrase_features,
    extract_temporal_organization_features,
    extract_targeted_separation_features,
    SR, CHUNK_DURATION, N_FFT, HOP_LENGTH,
)

# ─── optional microphone support ─────────────────────────────
try:
    import sounddevice as sd
    import soundfile as sf
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# ─── paths (cross-platform via pathlib) ──────────────────────
BASE_DIR            = Path(__file__).resolve().parent.parent
MODELS_DIR          = BASE_DIR / 'models'
COMPOSER_MODEL_PATH = MODELS_DIR / 'composer_model.pkl'
ERA_MODEL_PATH      = MODELS_DIR / 'era_model.pkl'
# ─── audio / feature constants ───────────────────────────────
FREQ_LOW_MAX   =  400.0
FREQ_MID_MAX   = 2000.0
FRAMES_PER_SEC = SR / HOP_LENGTH          # ≈ 43.1
RECORD_SECONDS = 30
EXCLUDED_FEATURES = set()

# ─── display maps ────────────────────────────────────────────
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

# ─── musical labels for features ─────────────────────────────
FEATURE_LABELS = {
    # Acoustic — timbre
    'mfcc_mean_0':               ('Overall energy',         'total spectral energy level'),
    'mfcc_mean_1':               ('Spectral slope',         'balance of bass vs treble energy'),
    'mfcc_mean_2':               ('Tonal colour',           'brightness vs warmth of the sound'),
    'mfcc_mean_3':               ('Spectral shape',         'curvature of the frequency envelope'),
    'mfcc_mean_4':               ('Harmonic texture',       'mid-range harmonic character'),
    'mfcc_mean_6':               ('Mid texture',            'character of mid-frequency content'),
    'mfcc_mean_7':               ('Timbral texture',        'fine-grained spectral texture'),
    'mfcc_mean_10':              ('Upper harmonics',        'character of upper harmonic content'),
    'mfcc_mean_11':              ('High harmonics',         'high-order spectral character'),
    'mfcc_mean_12':              ('Finest texture',         'highest-order timbral detail'),
    'mfcc_std_1':                ('Spectral variation',     'how much the spectral slope varies'),
    'mfcc_std_3':                ('Shape variation',        'variability in spectral curvature'),
    # Acoustic — dynamics
    'rms_mean':                  ('Loudness',               'average loudness of the piece'),
    'rms_std':                   ('Dynamic variation',      'how much the loudness fluctuates'),
    'dynamic_range':             ('Dynamic range',          'difference between loudest and quietest'),
    'loudness_slope':            ('Loudness trend',         'whether the piece grows louder or quieter'),
    'spectral_flux':             ('Spectral activity',      'how quickly the spectrum changes'),
    'zcr_mean':                  ('Noisiness',              'how noisy vs tonal the overall texture is'),
    # Acoustic — spectral
    'spectral_bandwidth':        ('Timbral richness',       'width of the harmonic spread'),
    'spectral_centroid':         ('Brightness',             'centre of mass of the frequency spectrum'),
    'spectral_rolloff':          ('Frequency ceiling',      'frequency below which 85% of energy falls'),
    'spectral_flatness':         ('Tone purity',            'how pure and tonal vs noisy the sound is'),
    'spectral_contrast_1':       ('Low contrast',           'spectral contrast in lower frequencies'),
    'spectral_contrast_2':       ('Mid contrast',           'spectral contrast in mid frequencies'),
    'spectral_contrast_6':       ('Brilliance contrast',    'sharpness of the highest frequency peaks'),
    'spectral_peak_count':       ('Polyphony density',      'number of simultaneous frequency peaks'),
    # Acoustic — rhythm / harmony / texture
    'polyphony_estimate':        ('Voice count',            'estimated number of simultaneous voices'),
    'hp_ratio':                  ('Harmonic purity',        'ratio of melodic to percussive content'),
    'tempo':                     ('Tempo',                  'estimated beats per minute'),
    'onset_rate':                ('Note density',           'how many notes are played per second'),
    'rhythm_regularity':         ('Rhythmic regularity',    'consistency of note timing'),
    'tempogram_entropy':         ('Rhythmic complexity',    'spread of rhythmic energy across tempos'),
    'tonal_instability':         ('Harmonic movement',      'how rapidly the harmony shifts'),
    'key_clarity':               ('Key strength',           'how strongly one key dominates'),
    'chromaticism':              ('Chromaticism',           'use of notes outside the home key'),
    'autocorr_peak':             ('Repetition',             'how strongly the piece repeats itself'),
    'self_similarity_peakiness': ('Structural contrast',    'variety between different sections'),
    # Stylistic — register
    'register_continuity_high':  ('High-register melody',   'how continuously the melody stays high — typical of Baroque solo writing'),
    'register_continuity_mid':   ('Mid-register activity',  'continuity of melodic activity in the middle register'),
    'register_balance':          ('Register balance',       'how evenly spread the music is across low, mid, and high registers'),
    'register_crossing_rate':    ('Voice crossing',         'how often voices cross — common in Bach counterpoint'),
    'band_activity_balance':     ('Band balance',           'how balanced the energy is across bass, mid, and treble bands'),
    # Stylistic — voice independence
    'band_correlation_low_mid':  ('Bass–mid coupling',      'how much the bass and mid voices move together'),
    'band_correlation_mid_high': ('Mid–treble coupling',    'how much the mid and treble voices move together'),
    'independence_index':        ('Voice independence',     'how independently the different voices move'),
    'anti_homophony':            ('Anti-homophony',         'tendency to avoid all voices moving in the same rhythm'),
    # Stylistic — Bach–Vivaldi targeted
    'harmonic_rhythm_rate':      ('Harmonic rhythm',        'how quickly chords change — faster in Vivaldi, slower in Bach'),
    'bass_melodic_complexity':   ('Bass complexity',        'how melodically active the bass line is — hallmark of Bach'),
    'voice_interval_mirroring':  ('Voice mirroring',        'how much bass and melody move in mirror motion — Baroque counterpoint'),
    'note_repetition_rate':      ('Note repetition',        'repeated note patterns — more common in Vivaldi sequences'),
    'sequence_periodicity':      ('Sequence regularity',    'how regular the harmonic sequence pattern is'),
    # Stylistic — phrase / imitation / motivic
    'phrase_boundary_sharpness': ('Phrase clarity',         'how sharply defined the phrase endings are'),
    'imitation_density':         ('Imitation',              'how often ideas are echoed between voices — characteristic of Bach fugues'),
    'horizontal_motion_ratio':   ('Melodic motion',         'proportion of melodic vs chordal movement'),
    # Targeted — Tchaikovsky / Vivaldi
    'dynamic_arc_variance':      ('Dynamic arc',            'variance of energy in 4-second windows — Tchaikovsky builds dramatic crescendos'),
    'chromatic_saturation':      ('Chromatic density',      'fraction of moments with many pitch classes active — higher in Romantic music'),
    'mid_register_gap':          ('Register gap',           'energy in bass + treble vs mid — Vivaldi violin + bass leaves a gap in the middle'),
}


# ─── helpers ─────────────────────────────────────────────────
def make_bar(value, width=14):
    n = max(0, min(width, int(round(value * width))))
    return '█' * n + '░' * (width - n)


def normalize_path(raw: str) -> str:
    p = raw.strip()
    if len(p) >= 2 and p[0] in ('"', "'") and p[-1] == p[0]:
        p = p[1:-1]
    p = p.replace('\\', '/')
    # Convert Windows drive letters → WSL paths (only when running on Linux/WSL)
    if sys.platform.startswith('linux') and len(p) >= 3 and p[1] == ':' and p[2] == '/':
        drive = p[0].lower()
        p = f'/mnt/{drive}/{p[3:]}'
    p = os.path.expanduser(p)
    p = os.path.expandvars(p)
    return p

# ─── new feature extraction (matches features2.csv) ──────────
def extract_supplement_features(y, S):
    """
    Compute the 3 supplemental features added in extract_features2.py.
    These must be present for the Phase 3 model (trained on 164 features).
    """
    out  = {}
    Sp   = S ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    # 1. dynamic_arc_variance
    seg_frames = max(1, int(4 * FRAMES_PER_SEC))
    rms_frames = np.sqrt(np.mean(Sp, axis=0))
    n_segs     = rms_frames.shape[0] // seg_frames
    if n_segs >= 2:
        seg_means = [float(np.mean(rms_frames[i * seg_frames:(i + 1) * seg_frames]))
                     for i in range(n_segs)]
        out['dynamic_arc_variance'] = float(np.var(seg_means))
    else:
        out['dynamic_arc_variance'] = 0.0

    # 2. chromatic_saturation
    try:
        chroma    = librosa.feature.chroma_stft(S=S, sr=SR, n_fft=N_FFT)
        frame_max = chroma.max(axis=0, keepdims=True) + 1e-8
        active    = (chroma / frame_max > 0.1).sum(axis=0)
        out['chromatic_saturation'] = float(np.mean(active >= 8))
    except Exception:
        out['chromatic_saturation'] = 0.5

    # 3. mid_register_gap
    try:
        low   = float(Sp[freqs < FREQ_LOW_MAX].sum())
        high  = float(Sp[freqs > FREQ_MID_MAX].sum())
        total = float(Sp.sum()) + 1e-10
        out['mid_register_gap'] = (low + high) / total
    except Exception:
        out['mid_register_gap'] = 0.5

    return out


# ─── audio processing ────────────────────────────────────────
def extract_chunk_features(y, sr, S):
    """Extract all 164 features from one 30-second chunk."""
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr, n_fft=N_FFT)
    row = {}
    # Acoustic (114 features)
    row.update(extract_mfcc_features(y, sr))
    row.update(extract_spectral_features(y, S, sr))        # note: y added
    row.update(extract_harmony_features(sr, chroma))
    row.update(extract_rhythm_features(y, sr))
    row.update(extract_dynamics_features(y, sr, S))
    row.update(extract_texture_features(y))
    row.update(extract_structure_features(chroma))
    row.update(extract_polyphony_features(S))
    # Stylistic (47 features)
    row.update(extract_horizontal_vertical_features(chroma))
    row.update(extract_register_features(S, sr))
    row.update(extract_voice_independence_features(S, sr))
    row.update(extract_melodic_motion_features(chroma))
    row.update(extract_imitation_features(chroma))
    row.update(extract_motivic_features(chroma))
    row.update(extract_phrase_features(y, sr, chroma))
    row.update(extract_temporal_organization_features(chroma, S))
    row.update(extract_targeted_separation_features(y, S, sr, chroma))
    # Supplement features2.csv (+3)
    row.update(extract_supplement_features(y, S))
    return row


def load_and_chunk(file_path: str):
    """Load an audio file and extract features from each 30-second window."""
    duration = librosa.get_duration(path=file_path)
    print(f"  Duration : {duration:.1f}s  →  ", end='', flush=True)

    chunks = []
    start  = 0.0

    while start + CHUNK_DURATION <= duration:
        y, sr = librosa.load(file_path, sr=SR,
                              offset=start, duration=CHUNK_DURATION)
        if len(y) >= SR * 5:
            S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
            chunks.append(extract_chunk_features(y, sr, S))
        start += CHUNK_DURATION

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
    """Convert list of feature dicts to numpy matrix, excluding training-dropped features."""
    feature_names = [k for k in chunks[0].keys() if k not in EXCLUDED_FEATURES]
    X = np.array([[chunk[k] for k in feature_names] for chunk in chunks])
    return X, feature_names


# ─── microphone recording ─────────────────────────────────────
def record_from_mic() -> str:
    """Record audio from microphone, save to temp WAV, return path."""
    print(f"\n  Recording for {RECORD_SECONDS} seconds.")
    print("  Play your music now — press Ctrl+C to stop early.\n")

    try:
        recording = sd.rec(
            int(RECORD_SECONDS * SR),
            samplerate=SR, channels=1, dtype='float32'
        )
        for i in range(RECORD_SECONDS):
            time.sleep(1)
            done  = '█' * (i + 1)
            left  = '░' * (RECORD_SECONDS - i - 1)
            print(f"  [{done}{left}] {i + 1}s / {RECORD_SECONDS}s", end='\r')
        sd.wait()
        print()

    except KeyboardInterrupt:
        sd.stop()
        print("\n  Stopped early.")

    # Trim trailing near-silence
    data = recording.squeeze()
    nonzero = np.flatnonzero(np.abs(data) > 1e-5)
    if len(nonzero) > 0:
        data = data[:nonzero[-1] + 1]

    if len(data) < SR * 10:
        raise ValueError("Recording too short — need at least 10 seconds of audio.")

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tmp.name, data, SR)
    print(f"  Saved recording ({len(data)/SR:.1f}s)")
    return tmp.name


# ─── display ─────────────────────────────────────────────────
def display_results(composer_probs, composer_classes, era_probs, era_classes):
    """Print era + composer probability bars and final verdict."""
    best_composer = composer_classes[np.argmax(composer_probs)]
    best_era      = era_classes[np.argmax(era_probs)]

    print()
    print("─" * 50)
    print("  ERA")
    for cls, prob in sorted(zip(era_classes, era_probs), key=lambda x: -x[1]):
        name = ERA_DISPLAY.get(cls, cls.capitalize())
        print(f"  {name:<12} {make_bar(prob)}  {prob * 100:.0f}%")

    print()
    print("  COMPOSER")
    for cls, prob in sorted(zip(composer_classes, composer_probs), key=lambda x: -x[1]):
        name = COMPOSER_LAST.get(cls, cls.capitalize())
        print(f"  {name:<12} {make_bar(prob)}  {prob * 100:.0f}%")

    print()
    era_str      = ERA_DISPLAY.get(best_era, best_era)
    composer_str = COMPOSER_FULL.get(best_composer, best_composer)
    print(f"  ► {composer_str}  [{era_str}]")
    print("─" * 50)


def display_feature_insights(composer_model, feature_names, X, top_n=15):
    """Show top features that drove the prediction in musical terms."""
    clf    = (composer_model.named_steps.get('gb')
               or composer_model.named_steps.get('rf'))
    scaler = composer_model.named_steps['scaler']

    importances  = clf.feature_importances_
    top_indices  = np.argsort(importances)[::-1][:top_n]

    X_scaled = scaler.transform(X)
    z_mean   = np.mean(X_scaled, axis=0)

    # Tag which family each top feature belongs to
    STYLISTIC = {
        'register_continuity_high', 'register_continuity_mid', 'register_balance',
        'register_crossing_rate', 'band_activity_balance', 'band_correlation_low_mid',
        'band_correlation_mid_high', 'independence_index', 'anti_homophony',
        'harmonic_rhythm_rate', 'bass_melodic_complexity', 'voice_interval_mirroring',
        'note_repetition_rate', 'sequence_periodicity', 'phrase_boundary_sharpness',
        'continuity_index', 'phrase_length_regularity', 'cadential_density',
        'imitation_density', 'chroma_lag_corr_1s', 'chroma_lag_corr_2s',
        'chroma_lag_corr_4s', 'transposition_invariant_recurrence', 'lag_profile_entropy',
        'horizontal_motion_ratio', 'chordal_jump_density', 'melodic_continuity',
        'stepwise_motion_ratio', 'leap_ratio', 'large_leap_ratio',
        'avg_interval_size', 'direction_balance', 'motion_variety', 'stasis_ratio',
        'dynamic_arc_variance', 'chromatic_saturation', 'mid_register_gap',
    }

    print(f"\n  TOP {top_n} CHARACTERISTICS THAT DROVE THIS RESULT")
    print("─" * 50)

    for rank, idx in enumerate(top_indices, 1):
        fname = feature_names[idx]
        z     = float(z_mean[idx])
        imp   = importances[idx]
        tag   = '★' if fname in STYLISTIC else ' '

        label, desc = FEATURE_LABELS.get(fname, (fname.replace('_', ' ').title(), ''))

        # Map z-score to level label
        if   z >  1.5: level = 'notably high'
        elif z >  0.5: level = 'above average'
        elif z > -0.5: level = 'around average'
        elif z > -1.5: level = 'below average'
        else:          level = 'notably low'

        b = make_bar((z + 3) / 6.0, width=14)

        print(f"\n  {rank:>2}. {tag} {label}")
        print(f"      {b}  {level}  (weight: {imp:.3f})")
        if desc:
            print(f"      ↳ {desc}")

    print(f"\n  ★ = stylistic / compositional feature")
    print("─" * 50)


# ─── main ────────────────────────────────────────────────────
def main():
    print()
    print("═" * 50)
    print("  AURALENS  —  Composer Identifier")
    print("═" * 50)

    # ── load models ───────────────────────────────────────────
    for path, name in [(COMPOSER_MODEL_PATH, 'composer'),
                       (ERA_MODEL_PATH, 'era')]:
        if not path.exists():
            print(f"\n  ERROR: {name} model not found at {path}")
            print("         Run train.py first.")
            sys.exit(1)

    composer_model = joblib.load(COMPOSER_MODEL_PATH)
    era_model      = joblib.load(ERA_MODEL_PATH)
    print("  Models loaded ✓")

    # ── input mode ───────────────────────────────────────────
    print()
    print("  How would you like to provide the audio?")
    print()
    print("  [1] Audio file  (mp3, wav, flac, ogg, m4a)")
    if MIC_AVAILABLE:
        print("  [2] Record from microphone")
    else:
        print("  [2] Record from microphone  "
              "(unavailable — pip install sounddevice soundfile)")
    print()

    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice in ('1', '2'):
            break
        print("  Please enter 1 or 2.")

    tmp_file = None

    # ── option 1: file ────────────────────────────────────────
    if choice == '1':
        print()
        print("  ┌─ TIP FOR NON-TECH USERS ──────────────────────┐")
        print("  │  Drag your audio file onto this window        │")
        print("  │  and press Enter — no need to type the path.  │")
        print("  │  Works on Windows, macOS, and Linux.          │")
        print("  └───────────────────────────────────────────────┘")
        print()
        print("  Supported formats: mp3  wav  flac  ogg  m4a")
        print()

        while True:
            raw = input("  > ").strip()
            if not raw:
                continue
            file_path = normalize_path(raw)
            if os.path.isfile(file_path):
                break
            print(f"\n  File not found: {file_path}")
            print("  Please try again.\n")

    # ── option 2: microphone ──────────────────────────────────
    else:
        if not MIC_AVAILABLE:
            print("\n  Microphone support requires:")
            print("    pip install sounddevice soundfile")
            sys.exit(1)

        try:
            file_path = record_from_mic()
            tmp_file  = file_path
        except Exception as e:
            print(f"\n  Recording failed: {e}")
            if 'device' in str(e).lower() or 'PortAudio' in str(e):
                print("\n  NOTE: Microphone is not supported in WSL.")
                print("  To use recording, run predict.py with native")
                print("  Windows Python (not WSL) or on Linux/macOS.")
            sys.exit(1)

    # ── run pipeline ─────────────────────────────────────────
    try:
        print("\n  [ Analysing... ]")
        chunks = load_and_chunk(file_path)
        X, feature_names = chunks_to_matrix(chunks)

        composer_probs   = np.mean(composer_model.predict_proba(X), axis=0)
        era_probs        = np.mean(era_model.predict_proba(X), axis=0)
        composer_classes = composer_model.classes_
        era_classes      = era_model.classes_

        display_results(composer_probs, composer_classes,
                        era_probs, era_classes)
        display_feature_insights(composer_model, feature_names, X, top_n=15)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    finally:
        # Clean up temp recording file if created
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


if __name__ == "__main__":
    main()