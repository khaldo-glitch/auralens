import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ─── settings ────────────────────────────────────────────────
SR             = 22050
CHUNK_DURATION = 30
N_FFT          = 2048
HOP_LENGTH     = 512

PROCESSED_DIR  = os.path.expanduser("~/auralens/data/processed")
RAW_DIR        = os.path.expanduser("~/auralens/data/raw")
OUTPUT_CSV     = os.path.expanduser("~/auralens/data/features.csv")

COMPOSERS      = ["bach", "vivaldi", "paganini", "tchaikovsky"]
ERA_MAP        = {
    "bach":        "baroque",
    "vivaldi":     "baroque",
    "paganini":    "romantic",
    "tchaikovsky": "romantic"
}

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# ─────────────────────────────────────────────────────────────


# ── CATEGORY 1: Timbre ───────────────────────────────────────
def extract_mfcc_features(y, sr):
    features = {}

    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(13):
        features[f'mfcc_mean_{i}']   = float(np.mean(mfcc[i]))
        features[f'mfcc_std_{i}']    = float(np.std(mfcc[i]))
        features[f'delta_mean_{i}']  = float(np.mean(delta[i]))
        features[f'delta2_mean_{i}'] = float(np.mean(delta2[i]))

    return features  # 52 features


# ── CATEGORY 2: Spectral shape ───────────────────────────────
# All librosa spectral shape functions expect a MAGNITUDE spectrogram
# when using S=. Our .npy files store magnitude, so we pass S directly.
# (chroma_stft is the exception — it expects power, handled separately.)
def extract_spectral_features(y, S, sr):
    features = {}

    features['spectral_centroid']  = float(np.mean(
        librosa.feature.spectral_centroid(S=S, sr=sr)))
    features['spectral_rolloff']   = float(np.mean(
        librosa.feature.spectral_rolloff(S=S, sr=sr)))
    features['spectral_bandwidth'] = float(np.mean(
        librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    features['spectral_flatness']  = float(np.mean(
        librosa.feature.spectral_flatness(S=S)))

    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    for i in range(contrast.shape[0]):
        features[f'spectral_contrast_{i}'] = float(np.mean(contrast[i]))

    return features  # 11 features


# ── CATEGORY 3: Harmony & tonality ──────────────────────────
def extract_harmony_features(sr, chroma):
    features = {}

    chroma_mean = np.mean(chroma, axis=1)
    chroma_std  = np.std(chroma,  axis=1)

    for i, v in enumerate(chroma_mean):
        features[f'chroma_mean_{i}'] = float(v)
    for i, v in enumerate(chroma_std):
        features[f'chroma_std_{i}'] = float(v)

    # low entropy = tonal (energy in few pitch classes)
    # high entropy = chromatic (energy spread across all 12)
    p = chroma_mean / (np.sum(chroma_mean) + 1e-10)
    p = np.maximum(p, 1e-10)
    p = p / np.sum(p)
    features['chroma_entropy'] = float(-np.sum(p * np.log2(p)))

    tonnetz      = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    for i, v in enumerate(tonnetz_mean):
        features[f'tonnetz_mean_{i}'] = float(v)

    major_scores = []
    minor_scores = []
    for shift in range(12):
        major_scores.append(
            np.corrcoef(chroma_mean, np.roll(MAJOR_PROFILE, shift))[0, 1])
        minor_scores.append(
            np.corrcoef(chroma_mean, np.roll(MINOR_PROFILE, shift))[0, 1])

    valid_major = [v for v in major_scores if not np.isnan(v)]
    valid_minor = [v for v in minor_scores if not np.isnan(v)]
    best_major  = float(max(valid_major)) if valid_major else 0.0
    best_minor  = float(max(valid_minor)) if valid_minor else 0.0

    features['key_clarity'] = max(best_major, best_minor)

    # clamp negatives to 0 — negative correlation means "not this key",
    # not "opposite mode"; clamping keeps key_mode in [0, 1]
    best_major_pos = max(best_major, 0.0)
    best_minor_pos = max(best_minor, 0.0)
    features['key_mode'] = best_major_pos / (
        best_major_pos + best_minor_pos + 1e-10)

    if best_major >= best_minor:
        clean = [v if not np.isnan(v) else -np.inf for v in major_scores]
    else:
        clean = [v if not np.isnan(v) else -np.inf for v in minor_scores]
    best_shift = int(np.argmax(clean))

    diatonic_mask = np.roll(
        np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool),
        best_shift)
    total_energy          = np.sum(chroma_mean) + 1e-10
    features['chromaticism'] = float(
        np.sum(chroma_mean[~diatonic_mask]) / total_energy)

    diffs = np.diff(tonnetz, axis=1)
    features['tonal_instability'] = float(
        np.mean(np.sqrt(np.sum(diffs ** 2, axis=0))))

    return features  # 35 features


# ── CATEGORY 4: Rhythm & tempo ───────────────────────────────
def extract_rhythm_features(y, sr):
    features = {}

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features['tempo'] = float(np.squeeze(tempo))

    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
    features['onset_rate'] = float(len(onsets) / CHUNK_DURATION)

    if len(onsets) > 1:
        onset_times = librosa.frames_to_time(
            onsets, sr=sr, hop_length=HOP_LENGTH)
        features['rhythm_regularity'] = float(np.std(np.diff(onset_times)))
    else:
        features['rhythm_regularity'] = 0.0

    tempogram  = librosa.feature.tempogram(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo_prof = np.mean(tempogram, axis=1)
    tempo_prof = tempo_prof / (np.sum(tempo_prof) + 1e-10)
    tempo_prof = np.maximum(tempo_prof, 1e-10)
    tempo_prof = tempo_prof / np.sum(tempo_prof)
    features['tempogram_entropy'] = float(
        -np.sum(tempo_prof * np.log2(tempo_prof)))

    # 1-frame tolerance — exact beat/onset frame alignment is unreliable
    if len(beats) > 0 and len(onsets) > 0:
        off_beat = sum(
            1 for o in onsets
            if not any(abs(o - b) <= 1 for b in beats))
        features['syncopation'] = float(off_beat / (len(onsets) + 1e-10))
    else:
        features['syncopation'] = 0.0

    return features  # 5 features


# ── CATEGORY 5: Dynamics & energy ───────────────────────────
def extract_dynamics_features(y, sr, S):
    features = {}

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std']  = float(np.std(rms))

    rms_max = float(np.max(rms))
    rms_min = float(np.min(rms))
    if rms_max < 1e-10:
        features['dynamic_range'] = 0.0
    else:
        features['dynamic_range'] = float(
            20 * np.log10((rms_max + 1e-10) / (rms_min + 1e-10)))

    x     = np.arange(len(rms), dtype=float)
    slope = np.polyfit(x, rms, 1)[0]
    features['loudness_slope'] = float(slope)

    S_diff = np.diff(S, axis=1)
    features['spectral_flux'] = float(np.mean(np.sum(S_diff ** 2, axis=0)))

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    features['zcr_mean'] = float(np.mean(zcr))

    return features  # 6 features


# ── CATEGORY 6: Texture ──────────────────────────────────────
def extract_texture_features(y):
    features = {}

    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic ** 2))
    p_energy = float(np.sum(percussive ** 2))
    features['hp_ratio'] = h_energy / (h_energy + p_energy + 1e-10)

    return features  # 1 feature


# ── CATEGORY 7: Structure ────────────────────────────────────
def extract_structure_features(chroma):
    features = {}

    n_frames = chroma.shape[1]
    seg_size = max(1, n_frames // 10)
    segments = [
        np.mean(chroma[:, i * seg_size:(i + 1) * seg_size], axis=1)
        for i in range(10)
    ]

    sim_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            num = np.dot(segments[i], segments[j])
            den = (np.linalg.norm(segments[i]) *
                   np.linalg.norm(segments[j]) + 1e-10)
            sim_matrix[i, j] = num / den

    off_diag = sim_matrix[~np.eye(10, dtype=bool)]
    features['self_similarity_peakiness'] = float(np.std(off_diag))

    chroma_sum = np.sum(chroma, axis=0)
    chroma_sum = chroma_sum - np.mean(chroma_sum)
    n          = len(chroma_sum)
    fft_result = np.fft.rfft(chroma_sum, n=2 * n)
    autocorr   = np.fft.irfft(fft_result * np.conj(fft_result))[:n]
    autocorr   = autocorr / (autocorr[0] + 1e-10)
    max_lag    = min(1000, n - 1)
    features['autocorr_peak'] = float(np.max(autocorr[1:max_lag]))

    return features  # 2 features


# ── CATEGORY 8: Polyphony ────────────────────────────────────
def extract_polyphony_features(S):
    features = {}

    mean_spec = np.mean(S, axis=1)
    is_peak   = ((mean_spec[1:-1] > mean_spec[:-2]) &
                 (mean_spec[1:-1] > mean_spec[2:]))
    features['spectral_peak_count'] = int(np.sum(is_peak))

    above_prev      = S[1:-1, :] > S[:-2, :]
    above_next      = S[1:-1, :] > S[2:, :]
    above_mean      = S[1:-1, :] > np.mean(S, axis=0, keepdims=True)
    peaks_per_frame = np.sum(above_prev & above_next & above_mean, axis=0)
    features['polyphony_estimate'] = float(np.mean(peaks_per_frame))

    return features  # 2 features


# ── MAIN EXTRACTION ──────────────────────────────────────────

def extract_all_features(npy_path, composer, mp3_files):
    S = np.load(npy_path)  # magnitude spectrogram, shape (1025, frames)

    basename = os.path.basename(npy_path)
    try:
        chunk_idx = int(basename.split('_chunk_')[1].split('.')[0])
        mp3_base  = basename.split('_chunk_')[0]
    except (IndexError, ValueError):
        return None

    offset    = chunk_idx * CHUNK_DURATION
    mp3_dir   = os.path.join(RAW_DIR, composer)

    # preprocessing truncated filenames to 50 chars — match on the same basis
    mp3_match = next(
        (f for f in mp3_files
         if os.path.splitext(f)[0].replace(' ', '_')[:50] == mp3_base),
        None)

    if mp3_match is None:
        return None

    mp3_path = os.path.join(mp3_dir, mp3_match)
    y, sr    = librosa.load(mp3_path, sr=SR,
                             offset=offset,
                             duration=CHUNK_DURATION)

    if len(y) < SR * 5:
        return None

    # chroma_stft expects power — convert magnitude → power with S**2
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr, n_fft=N_FFT)

    row = {}
    row.update(extract_mfcc_features(y, sr))
    row.update(extract_spectral_features(y, S, sr))
    row.update(extract_harmony_features(sr, chroma))
    row.update(extract_rhythm_features(y, sr))
    row.update(extract_dynamics_features(y, sr, S))
    row.update(extract_texture_features(y))
    row.update(extract_structure_features(chroma))
    row.update(extract_polyphony_features(S))

    row['composer'] = composer
    row['era']      = ERA_MAP[composer]

    return row


# ── MAIN LOOP ────────────────────────────────────────────────

def main():
    all_rows = []

    for composer in COMPOSERS:
        proc_dir  = os.path.join(PROCESSED_DIR, composer)
        mp3_dir   = os.path.join(RAW_DIR, composer)

        npy_files = sorted(
            f for f in os.listdir(proc_dir) if f.endswith('.npy'))
        mp3_files = [f for f in os.listdir(mp3_dir) if f.endswith('.mp3')]

        print(f"\n── {composer.upper()} ({len(npy_files)} chunks) ──")

        for filename in tqdm(npy_files):
            npy_path = os.path.join(proc_dir, filename)
            try:
                row = extract_all_features(npy_path, composer, mp3_files)
                if row is not None:
                    all_rows.append(row)
            except Exception as e:
                print(f"  ✗ {filename}: {e}")

        print(f"  rows so far: {len(all_rows)}")

    if not all_rows:
        print("No rows extracted — check paths and filenames.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\ndone. saved {len(df)} rows x {len(df.columns)} columns")
    print(f"saved to: {OUTPUT_CSV}")

    feature_cols = [c for c in df.columns if c not in ('composer', 'era')]
    print(f"\nfeature count: {len(feature_cols)}")
    print(f"  mfcc (mean+std+delta+delta2 × 13):                52")
    print(f"  spectral (centroid/rolloff/bandwidth/flat/contrast): 11")
    print(f"  harmony (chroma+entropy+tonnetz+key+chroma):       35")
    print(f"  rhythm (tempo/onsets/regularity/tempogram/sync):    5")
    print(f"  dynamics (rms/range/slope/flux/zcr):                6")
    print(f"  texture (hp_ratio):                                  1")
    print(f"  structure (self_similarity/autocorr):               2")
    print(f"  polyphony (peak_count/estimate):                     2")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  total:                                             114")


if __name__ == "__main__":
    main()
