import librosa
import numpy as np
import pandas as pd
import os
from collections import Counter
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

# frequency band boundaries for register analysis
FREQ_LOW_MIN  =   80.0
FREQ_LOW_MAX  =  400.0
FREQ_MID_MAX  = 2000.0
FREQ_HIGH_MAX = 8000.0

# frame rate: how many frames per second at our settings
FRAMES_PER_SEC = SR / HOP_LENGTH   # ≈ 43.1
# ─────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════
#  EXISTING ACOUSTIC FEATURES — unchanged
# ════════════════════════════════════════════════════════════

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


def extract_harmony_features(sr, chroma):
    features = {}
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std  = np.std(chroma,  axis=1)
    for i, v in enumerate(chroma_mean):
        features[f'chroma_mean_{i}'] = float(v)
    for i, v in enumerate(chroma_std):
        features[f'chroma_std_{i}'] = float(v)
    p = chroma_mean / (np.sum(chroma_mean) + 1e-10)
    p = np.maximum(p, 1e-10)
    p = p / np.sum(p)
    features['chroma_entropy'] = float(-np.sum(p * np.log2(p)))
    tonnetz      = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    for i, v in enumerate(tonnetz_mean):
        features[f'tonnetz_mean_{i}'] = float(v)
    major_scores, minor_scores = [], []
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
    best_major_pos = max(best_major, 0.0)
    best_minor_pos = max(best_minor, 0.0)
    features['key_mode'] = best_major_pos / (
        best_major_pos + best_minor_pos + 1e-10)
    if best_major >= best_minor:
        clean = [v if not np.isnan(v) else -np.inf for v in major_scores]
    else:
        clean = [v if not np.isnan(v) else -np.inf for v in minor_scores]
    best_shift    = int(np.argmax(clean))
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
    if len(beats) > 0 and len(onsets) > 0:
        off_beat = sum(
            1 for o in onsets
            if not any(abs(o - b) <= 1 for b in beats))
        features['syncopation'] = float(off_beat / (len(onsets) + 1e-10))
    else:
        features['syncopation'] = 0.0
    return features  # 5 features


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
            min(20 * np.log10((rms_max + 1e-10) / (rms_min + 1e-10)), 80.0))
    x     = np.arange(len(rms), dtype=float)
    slope = np.polyfit(x, rms, 1)[0]
    features['loudness_slope'] = float(slope)
    S_diff = np.diff(S, axis=1)
    features['spectral_flux'] = float(np.mean(np.sum(S_diff ** 2, axis=0)))
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    return features  # 6 features


def extract_texture_features(y):
    features = {}
    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic ** 2))
    p_energy = float(np.sum(percussive ** 2))
    features['hp_ratio'] = h_energy / (h_energy + p_energy + 1e-10)
    return features  # 1 feature


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


# ════════════════════════════════════════════════════════════
#  NEW: STRUCTURAL / STYLISTIC FEATURES
#  These approximate HOW the music is written, not how it sounds.
#  All features are scalars aggregated over the 30-second chunk.
# ════════════════════════════════════════════════════════════

def _get_dominant_chroma_sequence(chroma):
    """
    Returns the sequence of dominant pitch class (0-11) per frame,
    and semitone intervals between consecutive frames wrapped to [-6, 6].
    This is our proxy for the 'melodic surface' of the chunk.
    """
    dominant = np.argmax(chroma, axis=0).astype(int)   # shape: (n_frames,)
    intervals = np.diff(dominant)
    # wrap to shortest path on chromatic circle
    intervals = ((intervals + 6) % 12) - 6
    return dominant, intervals


def _get_band_energies(S, sr):
    """
    Split the magnitude spectrogram into three frequency registers:
      low  =  80–400  Hz  (bass / cello / left hand)
      mid  = 400–2000 Hz  (tenor / alto / melody)
      high = 2000–8000 Hz (soprano / violin / upper voices)
    Returns energy envelope per band per frame.
    """
    freqs     = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    low_mask  = (freqs >= FREQ_LOW_MIN)  & (freqs < FREQ_LOW_MAX)
    mid_mask  = (freqs >= FREQ_LOW_MAX)  & (freqs < FREQ_MID_MAX)
    high_mask = (freqs >= FREQ_MID_MAX)  & (freqs < FREQ_HIGH_MAX)
    low_e  = np.sum(S[low_mask,  :], axis=0)
    mid_e  = np.sum(S[mid_mask,  :], axis=0)
    high_e = np.sum(S[high_mask, :], axis=0)
    return low_e, mid_e, high_e


# ── Family 1: Horizontal vs Vertical Motion ──────────────────

def extract_horizontal_vertical_features(chroma):
    features = {}
    _, intervals = _get_dominant_chroma_sequence(chroma)

    features['horizontal_motion_ratio'] = float(
        np.mean(np.abs(intervals) <= 1))

    chroma_diff   = np.abs(np.diff(chroma, axis=1))
    bins_changed  = np.mean(chroma_diff > 0.05, axis=0)
    features['chordal_jump_density'] = float(np.mean(bins_changed > 0.5))

    norms = np.linalg.norm(chroma, axis=0) + 1e-10
    dots  = np.sum(chroma[:, :-1] * chroma[:, 1:], axis=0)
    cos_sim = dots / (norms[:-1] * norms[1:])
    features['melodic_continuity'] = float(np.mean(cos_sim))

    return features  # 3 features


# ── Family 2: Voice Persistence & Register Behavior ──────────

def extract_register_features(S, sr):
    features = {}
    low_e, mid_e, high_e = _get_band_energies(S, sr)

    mean_low  = float(np.mean(low_e))  + 1e-10
    mean_mid  = float(np.mean(mid_e))  + 1e-10
    mean_high = float(np.mean(high_e)) + 1e-10

    features['register_balance'] = mean_high / mean_low

    def lag1_autocorr(x):
        if len(x) < 2:
            return 0.0
        c = np.corrcoef(x[:-1], x[1:])[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    features['register_continuity_low']  = lag1_autocorr(low_e)
    features['register_continuity_mid']  = lag1_autocorr(mid_e)
    features['register_continuity_high'] = lag1_autocorr(high_e)

    band_stack    = np.stack([low_e, mid_e, high_e], axis=0)
    dominant_band = np.argmax(band_stack, axis=0)
    crossings     = np.sum(np.diff(dominant_band) != 0)
    features['register_crossing_rate'] = float(
        crossings / (len(dominant_band) - 1 + 1e-10))

    means = np.array([mean_low, mean_mid, mean_high])
    features['band_activity_balance'] = float(
        np.std(means) / (np.sum(means) + 1e-10))

    return features  # 6 features


# ── Family 3: True Polyphony & Voice Independence ─────────────

def extract_voice_independence_features(S, sr):
    features = {}
    low_e, mid_e, high_e = _get_band_energies(S, sr)

    def safe_corr(a, b):
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    c_lm = safe_corr(low_e, mid_e)
    c_lh = safe_corr(low_e, high_e)
    c_mh = safe_corr(mid_e, high_e)

    features['band_correlation_low_mid']  = c_lm
    features['band_correlation_low_high'] = c_lh
    features['band_correlation_mid_high'] = c_mh

    features['independence_index'] = float(
        np.mean([1 - abs(c_lm), 1 - abs(c_lh), 1 - abs(c_mh)]))

    band_stack = np.stack([low_e, mid_e, high_e], axis=0)
    frame_sum  = np.sum(band_stack, axis=0) + 1e-10
    band_norm  = band_stack / frame_sum
    frame_std  = np.std(band_norm, axis=0)
    features['anti_homophony'] = float(np.mean(frame_std > 0.15))

    return features  # 5 features


# ── Family 4: Melodic Motion Grammar ─────────────────────────

def extract_melodic_motion_features(chroma):
    features = {}
    _, intervals = _get_dominant_chroma_sequence(chroma)

    if len(intervals) == 0:
        for key in ['stepwise_motion_ratio', 'leap_ratio', 'large_leap_ratio',
                    'avg_interval_size', 'direction_balance',
                    'motion_variety', 'stasis_ratio']:
            features[key] = 0.0
        return features

    abs_intervals = np.abs(intervals)

    features['stepwise_motion_ratio'] = float(np.mean(abs_intervals <= 2))
    features['leap_ratio']            = float(np.mean(abs_intervals > 4))
    features['large_leap_ratio']      = float(np.mean(abs_intervals > 7))
    features['avg_interval_size']     = float(np.mean(abs_intervals))

    ascending  = float(np.mean(intervals > 0))
    descending = float(np.mean(intervals < 0))
    features['direction_balance'] = ascending / (ascending + descending + 1e-10)

    features['stasis_ratio'] = float(np.mean(intervals == 0))

    counts = np.bincount(abs_intervals, minlength=7).astype(float)
    counts = counts / (counts.sum() + 1e-10)
    counts = np.maximum(counts, 1e-10)
    counts = counts / counts.sum()
    features['motion_variety'] = float(-np.sum(counts * np.log2(counts)))

    return features  # 7 features


# ── Family 5: Imitation & Counterpoint Proxies ───────────────

def extract_imitation_features(chroma):
    features = {}
    n_frames = chroma.shape[1]

    chroma_sum = np.sum(chroma, axis=0)
    chroma_sum = chroma_sum - np.mean(chroma_sum)

    def lag_corr(signal, lag_frames):
        if lag_frames >= len(signal):
            return 0.0
        c = np.corrcoef(signal[:-lag_frames], signal[lag_frames:])[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    lag_1s = max(1, int(1.0 * FRAMES_PER_SEC))
    lag_2s = max(1, int(2.0 * FRAMES_PER_SEC))
    lag_4s = max(1, int(4.0 * FRAMES_PER_SEC))

    features['chroma_lag_corr_1s'] = lag_corr(chroma_sum, lag_1s)
    features['chroma_lag_corr_2s'] = lag_corr(chroma_sum, lag_2s)
    features['chroma_lag_corr_4s'] = lag_corr(chroma_sum, lag_4s)

    half_s  = max(1, int(0.5 * FRAMES_PER_SEC))
    lag_corrs = [lag_corr(chroma_sum, l)
                 for l in range(half_s, lag_4s + 1)]
    features['imitation_density'] = float(max(lag_corrs)) if lag_corrs else 0.0

    lag_arr = np.array(lag_corrs)
    lag_abs = np.abs(lag_arr) + 1e-10
    lag_abs = lag_abs / lag_abs.sum()
    features['lag_profile_entropy'] = float(
        -np.sum(lag_abs * np.log2(lag_abs)))

    if n_frames > 2:
        step = max(1, n_frames // 200)
        c_sub = chroma[:, ::step]
        n_sub = c_sub.shape[1]
        best_matches = []
        for t in range(n_sub):
            frame = c_sub[:, t]
            best = 0.0
            for shift in range(12):
                shifted = np.roll(frame, shift)
                dots = c_sub.T @ shifted
                norms = np.linalg.norm(c_sub, axis=0) * np.linalg.norm(shifted) + 1e-10
                sims  = dots / norms
                sims[t] = 0.0
                best = max(best, float(np.max(sims)))
            best_matches.append(best)
        features['transposition_invariant_recurrence'] = float(
            np.mean(best_matches))
    else:
        features['transposition_invariant_recurrence'] = 0.0

    return features  # 6 features


# ── Family 6: Motivic Development ────────────────────────────

def extract_motivic_features(chroma):
    features = {}
    _, intervals = _get_dominant_chroma_sequence(chroma)

    if len(intervals) < 6:
        for key in ['ngram3_repetition_ratio', 'ngram4_repetition_ratio',
                    'ngram5_repetition_ratio', 'motive_entropy_3',
                    'motive_entropy_4', 'top_motive_dominance']:
            features[key] = 0.0
        return features

    interval_list = intervals.tolist()

    def ngram_stats(seq, n):
        grams  = [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]
        counts = Counter(grams)
        total  = len(grams)
        if total == 0:
            return 0.0, 0.0
        repeated = sum(1 for c in counts.values() if c > 1)
        rep_ratio = repeated / len(counts)
        probs = np.array(list(counts.values()), dtype=float) / total
        probs = np.maximum(probs, 1e-10)
        probs = probs / probs.sum()
        entropy = float(-np.sum(probs * np.log2(probs)))
        return rep_ratio, entropy

    rep3, ent3 = ngram_stats(interval_list, 3)
    rep4, ent4 = ngram_stats(interval_list, 4)
    rep5, _    = ngram_stats(interval_list, 5)

    features['ngram3_repetition_ratio'] = rep3
    features['ngram4_repetition_ratio'] = rep4
    features['ngram5_repetition_ratio'] = rep5
    features['motive_entropy_3']        = ent3
    features['motive_entropy_4']        = ent4

    grams3  = [tuple(interval_list[i:i+3]) for i in range(len(interval_list)-2)]
    counts3 = Counter(grams3)
    total3  = len(grams3)
    if total3 > 0 and len(counts3) > 0:
        top5_count = sum(c for _, c in counts3.most_common(5))
        features['top_motive_dominance'] = float(top5_count / total3)
    else:
        features['top_motive_dominance'] = 0.0

    return features  # 6 features


# ── Family 7: Phrase & Architectural Behavior ─────────────────

def extract_phrase_features(y, sr, chroma):
    features = {}

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    rms_diff       = np.diff(rms)
    threshold      = np.std(rms_diff) * 1.5
    boundaries     = np.where(rms_diff < -threshold)[0]
    features['phrase_boundary_sharpness'] = float(
        np.mean(np.abs(rms_diff[boundaries])) if len(boundaries) > 0 else 0.0)

    n_drops = len(boundaries)
    features['continuity_index'] = float(
        1.0 - n_drops / (len(rms) + 1e-10))

    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH,
                                         delta=0.07)
    if len(onsets) > 3:
        onset_times = librosa.frames_to_time(onsets, sr=sr,
                                              hop_length=HOP_LENGTH)
        ioi         = np.diff(onset_times)
        long_ioi    = ioi[ioi > 1.0]
        features['phrase_length_regularity'] = float(
            np.std(long_ioi)) if len(long_ioi) > 1 else 0.0
    else:
        features['phrase_length_regularity'] = 0.0

    chroma_mean  = np.mean(chroma, axis=1)
    major_scores = [
        np.corrcoef(chroma_mean, np.roll(MAJOR_PROFILE, s))[0, 1]
        for s in range(12)
    ]
    major_scores = [v if not np.isnan(v) else 0.0 for v in major_scores]
    best_key     = int(np.argmax(major_scores))

    tonic_idx   = best_key % 12
    dominant_idx = (best_key + 7) % 12

    frame_tonic_strength   = chroma[tonic_idx, :]
    frame_dominant_strength = chroma[dominant_idx, :]
    dominant_chroma = np.argmax(chroma, axis=0)
    cadential_frames = np.sum(
        (dominant_chroma == tonic_idx) | (dominant_chroma == dominant_idx))
    features['cadential_density'] = float(
        cadential_frames / (chroma.shape[1] + 1e-10))

    return features  # 4 features


# ── Family 8: Temporal Organization ──────────────────────────

def extract_temporal_organization_features(chroma, S):
    features = {}

    dominant, _ = _get_dominant_chroma_sequence(chroma)
    dominant_f  = dominant.astype(float) - np.mean(dominant)
    lag_8s      = max(1, int(8.0 * FRAMES_PER_SEC))
    if len(dominant_f) > lag_8s + 1:
        c = np.corrcoef(dominant_f[:-lag_8s], dominant_f[lag_8s:])[0, 1]
        features['pitch_motion_autocorr_long'] = float(c) if not np.isnan(c) else 0.0
    else:
        features['pitch_motion_autocorr_long'] = 0.0

    n_frames    = chroma.shape[1]
    window_size = max(1, int(3.0 * FRAMES_PER_SEC))
    key_scores  = []
    for start in range(0, n_frames - window_size, window_size // 2):
        window_chroma = np.mean(
            chroma[:, start:start + window_size], axis=1)
        scores = [
            np.corrcoef(window_chroma, np.roll(MAJOR_PROFILE, s))[0, 1]
            for s in range(12)
        ]
        scores = [v if not np.isnan(v) else 0.0 for v in scores]
        key_scores.append(max(scores))
    features['tonal_center_stability'] = float(
        np.std(key_scores)) if len(key_scores) > 1 else 0.0

    rms        = librosa.feature.rms(y=None, S=S, hop_length=HOP_LENGTH)[0]
    rms_norm   = rms - np.mean(rms)
    rms_fft    = np.abs(np.fft.rfft(rms_norm))
    if len(rms_fft) > 1:
        features['activity_periodicity'] = float(
            np.max(rms_fft[1:]) / (np.sum(rms_fft[1:]) + 1e-10))
    else:
        features['activity_periodicity'] = 0.0

    n_t         = S.shape[1]
    frame_energy = np.mean(S, axis=0)
    x            = np.arange(n_t, dtype=float)
    slope        = np.polyfit(x, frame_energy, 1)[0]
    features['density_trend'] = float(slope)

    third = max(1, n_frames // 3)
    first_chroma = np.mean(chroma[:, :third],       axis=1)
    last_chroma  = np.mean(chroma[:, -third:],      axis=1)
    num  = np.dot(first_chroma, last_chroma)
    den  = (np.linalg.norm(first_chroma) *
            np.linalg.norm(last_chroma) + 1e-10)
    features['long_range_chroma_consistency'] = float(num / den)

    return features  # 5 features


# ── Family 9: Targeted Bach–Vivaldi Separation ───────────────

def extract_targeted_separation_features(y, S, sr, chroma):
    """
    5 features designed specifically to separate Bach from Vivaldi.
    Helpers _get_band_energies and _get_dominant_chroma_sequence are
    already defined above. All constants (SR, N_FFT, etc.) reused.
    """
    features = {}

    # ── pre-compute piptrack voices once, reused by features 1 and 3 ──
    def _piptrack_dominant(fmin, fmax):
        try:
            p, m  = librosa.piptrack(S=S, sr=sr, fmin=fmin, fmax=fmax,
                                      hop_length=HOP_LENGTH)
            n_f   = m.shape[1]
            idx   = np.argmax(m, axis=0)
            hz    = p[idx, np.arange(n_f)]
            valid = hz > 1.0
            midi  = np.where(valid,
                             librosa.hz_to_midi(np.maximum(hz, 1.0)),
                             np.nan)
            return midi, valid
        except Exception:
            return np.array([np.nan]), np.array([False])

    low_midi,  low_valid  = _piptrack_dominant(80,   400)
    high_midi, high_valid = _piptrack_dominant(2000, 8000)

    # ── Feature 1: bass_melodic_complexity ───────────────────
    # Measures how actively the bass voice moves melodically.
    # Bach's bass lines are highly contrapuntal — they carry independent
    # melodic content and change pitch frequently (walking bass, fugue subjects).
    # Vivaldi's bass is more static: basso continuo holds pedal tones or
    # repeats simple ostinato patterns. Higher value → more active bass.
    try:
        if low_valid.sum() < 2:
            features['bass_melodic_complexity'] = 0.0
        else:
            intervals   = np.diff(low_midi)
            valid_pairs = low_valid[:-1] & low_valid[1:]
            active      = (np.abs(intervals) > 1.0) & valid_pairs
            low_e, _, _ = _get_band_energies(S, sr)
            low_w       = low_e[:len(active)]
            low_w       = low_w / (low_w.sum() + 1e-10)
            result      = float(np.sum(active.astype(float) * low_w))
            features['bass_melodic_complexity'] = (
                result if not np.isnan(result) else 0.0)
    except Exception:
        features['bass_melodic_complexity'] = 0.0

    # ── Feature 2: harmonic_rhythm_rate ──────────────────────
    # Counts how many times per second the harmony changes substantially.
    # Bach's counterpoint produces frequent harmonic micro-changes as
    # independent voices create passing harmonies.
    # Vivaldi's block-chord style changes harmony less often.
    # Unit: changes per second.
    try:
        norms    = np.linalg.norm(chroma, axis=0) + 1e-10
        dots     = np.sum(chroma[:, :-1] * chroma[:, 1:], axis=0)
        cos_dist = 1.0 - dots / (norms[:-1] * norms[1:])
        changes  = float(np.sum(cos_dist > 0.15))
        features['harmonic_rhythm_rate'] = changes / CHUNK_DURATION
    except Exception:
        features['harmonic_rhythm_rate'] = 0.0

    # ── Feature 3: voice_interval_mirroring ──────────────────
    # Pearson correlation between the interval sequence of the bass voice
    # (80–400 Hz) and the soprano voice (2000–8000 Hz).
    # In Bach counterpoint, bass and soprano move independently → low / negative
    # correlation (contrary motion is a hallmark of good counterpoint).
    # In Vivaldi's homophony, all voices tend to move in the same direction
    # → high positive correlation.
    try:
        low_iv  = np.diff(low_midi)
        high_iv = np.diff(high_midi)
        both    = (low_valid[:-1]  & low_valid[1:] &
                   high_valid[:-1] & high_valid[1:])

        if both.sum() < 10:
            features['voice_interval_mirroring'] = 0.0
        else:
            li   = low_iv[both]
            hi   = high_iv[both]
            li   = ((li + 6) % 12) - 6
            hi   = ((hi + 6) % 12) - 6
            corr = np.corrcoef(li, hi)[0, 1]
            features['voice_interval_mirroring'] = (
                float(corr) if not np.isnan(corr) else 0.0)
    except Exception:
        features['voice_interval_mirroring'] = 0.0

    # ── Feature 4: note_repetition_rate ──────────────────────
    # Fraction of frames that belong to a run of 3 or more consecutive
    # identical dominant pitch classes.
    # Vivaldi uses repeated-note figures, ostinatos, and scalar sequences
    # with repeated pitches far more than Bach.
    # Bach's melodic lines tend to be more varied and through-composed.
    try:
        dominant, _ = _get_dominant_chroma_sequence(chroma)
        n = len(dominant)
        if n < 3:
            features['note_repetition_rate'] = 0.0
        else:
            frames_in_runs = 0
            i = 0
            while i < n:
                j = i + 1
                while j < n and dominant[j] == dominant[i]:
                    j += 1
                if (j - i) >= 3:
                    frames_in_runs += (j - i)
                i = j
            features['note_repetition_rate'] = float(frames_in_runs / n)
    except Exception:
        features['note_repetition_rate'] = 0.0

    # ── Feature 5: sequence_periodicity ──────────────────────
    # Strength of short-range periodicity (0.25–2 s) in harmonic activity.
    # Vivaldi is famous for sequential writing — the same melodic pattern
    # transposed up or down repeatedly — which creates strong periodic pulses
    # in the harmonic-change signal.
    # Bach's development is less mechanically periodic.
    # Returns the mean of the top-3 autocorrelation values in that lag range.
    try:
        chroma_diff = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)
        if len(chroma_diff) < 4:
            features['sequence_periodicity'] = 0.0
        else:
            signal  = chroma_diff - np.mean(chroma_diff)
            lag_min = max(1, int(0.25 * FRAMES_PER_SEC))
            lag_max = min(int(2.0  * FRAMES_PER_SEC), len(signal) - 1)
            if lag_min >= lag_max:
                features['sequence_periodicity'] = 0.0
            else:
                autocorrs = []
                for lag in range(lag_min, lag_max + 1):
                    c = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
                    autocorrs.append(float(c) if not np.isnan(c) else 0.0)
                top3 = sorted(autocorrs, reverse=True)[:3]
                features['sequence_periodicity'] = float(np.mean(top3))
    except Exception:
        features['sequence_periodicity'] = 0.0

    return features  # 5 features


# ════════════════════════════════════════════════════════════
#  MAIN EXTRACTION FUNCTION
# ════════════════════════════════════════════════════════════

def extract_all_features(npy_path, composer, mp3_files):
    S = np.load(npy_path)  # magnitude spectrogram (1025, n_frames)

    basename = os.path.basename(npy_path)
    try:
        chunk_idx = int(basename.split('_chunk_')[1].split('.')[0])
        mp3_base  = basename.split('_chunk_')[0]
    except (IndexError, ValueError):
        return None

    offset    = chunk_idx * CHUNK_DURATION
    mp3_dir   = os.path.join(RAW_DIR, composer)
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

    # chroma_stft expects power spectrogram
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr, n_fft=N_FFT)

    row = {}

    # ── existing acoustic features ────────────────────────────
    row.update(extract_mfcc_features(y, sr))
    row.update(extract_spectral_features(y, S, sr))
    row.update(extract_harmony_features(sr, chroma))
    row.update(extract_rhythm_features(y, sr))
    row.update(extract_dynamics_features(y, sr, S))
    row.update(extract_texture_features(y))
    row.update(extract_structure_features(chroma))
    row.update(extract_polyphony_features(S))

    # ── new structural / stylistic features ───────────────────
    row.update(extract_horizontal_vertical_features(chroma))
    row.update(extract_register_features(S, sr))
    row.update(extract_voice_independence_features(S, sr))
    row.update(extract_melodic_motion_features(chroma))
    row.update(extract_imitation_features(chroma))
    row.update(extract_motivic_features(chroma))
    row.update(extract_phrase_features(y, sr, chroma))
    row.update(extract_temporal_organization_features(chroma, S))

    # ── targeted Bach–Vivaldi separation features ─────────────
    row.update(extract_targeted_separation_features(y, S, sr, chroma))

    row['source_file'] = mp3_base
    row['composer']    = composer
    row['era']         = ERA_MAP[composer]

    return row


# ════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════

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
                print(f"  x {filename}: {e}")

        print(f"  rows so far: {len(all_rows)}")

    if not all_rows:
        print("No rows extracted — check paths and filenames.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    feature_cols = [c for c in df.columns
                    if c not in ('composer', 'era', 'source_file', 'key_mode')]
    print(f"\ndone. {len(df)} rows x {len(feature_cols)} features")
    print(f"saved to: {OUTPUT_CSV}")

    # feature count breakdown
    print(f"\nfeature breakdown:")
    print(f"  existing acoustic features:       114")
    print(f"  horizontal / vertical motion:       3")
    print(f"  register behavior:                  6")
    print(f"  voice independence:                 5")
    print(f"  melodic motion grammar:             7")
    print(f"  imitation / counterpoint proxies:   6")
    print(f"  motivic development:                6")
    print(f"  phrase / architecture:              4")
    print(f"  temporal organization:              5")
    print(f"  targeted Bach–Vivaldi separation:   5")
    print(f"  ─────────────────────────────────────")
    print(f"  total new features:                47")
    print(f"  total all features:               161")


if __name__ == "__main__":
    main()