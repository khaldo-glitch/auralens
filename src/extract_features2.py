#!/usr/bin/env python3
"""
extract_features2.py — Targeted supplement to features.csv

Reads pre-chunked .npy files from ~/auralens/data/processed/{composer}/
Adds 3 new features without re-chunking or touching extract_features.py:
  - dynamic_arc_variance  : long-arc dynamics  → Tchaikovsky vs Bach
  - chromatic_saturation  : chromatic density  → Tchaikovsky vs Bach
  - mid_register_gap      : melody+bass gap    → Vivaldi vs Bach
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ─── settings (must match extract_features.py) ───────────────
SR            = 22050
N_FFT         = 2048
HOP_LENGTH    = 512
FREQ_LOW_MAX  =  400.0
FREQ_MID_MAX  = 2000.0
FRAMES_PER_SEC = SR / HOP_LENGTH

PROCESSED_DIR = os.path.expanduser('~/auralens/data/processed')
DATA_CSV      = os.path.expanduser('~/auralens/data/features.csv')
OUT_CSV       = os.path.expanduser('~/auralens/data/features2.csv')
# ─────────────────────────────────────────────────────────────

FRAMES_PER_SEC = SR / HOP_LENGTH   # ≈ 43.07

def extract_new_features(S):
    """
    S : pre-computed magnitude spectrogram, shape (1025, 1292)
    All 3 features derived directly — no audio loading needed.
    """
    out  = {}
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    Sp    = S ** 2   # power spectrogram

    # ── 1. dynamic_arc_variance ───────────────────────────────
    # RMS energy per frame → group into 4-second windows → variance of means.
    # Tchaikovsky has wide crescendo/decrescendo arcs; Bach stays steady.
    seg_frames = max(1, int(4 * FRAMES_PER_SEC))   # ≈ 172 frames per 4 s
    rms_frames = np.sqrt(np.mean(Sp, axis=0))       # shape (n_frames,)
    n_segs     = rms_frames.shape[0] // seg_frames
    if n_segs >= 2:
        seg_means = [float(np.mean(rms_frames[i*seg_frames:(i+1)*seg_frames]))
                     for i in range(n_segs)]
        out['dynamic_arc_variance'] = float(np.var(seg_means))
    else:
        out['dynamic_arc_variance'] = 0.0

    # ── 2. chromatic_saturation ───────────────────────────────
    # Fraction of frames where ≥ 8 of 12 pitch classes are active.
    # Tchaikovsky is chromatic (8–12 active); Bach is diatonic (5–7).
    try:
        chroma    = librosa.feature.chroma_stft(S=S, sr=SR, n_fft=N_FFT)
        frame_max = chroma.max(axis=0, keepdims=True) + 1e-8
        active    = (chroma / frame_max > 0.1).sum(axis=0)
        out['chromatic_saturation'] = float(np.mean(active >= 8))
    except Exception:
        out['chromatic_saturation'] = 0.5

    # ── 3. mid_register_gap ───────────────────────────────────
    # (low-band + high-band) / total power.
    # Vivaldi: violin (high) + bass (low), gap in middle → high ratio.
    # Bach:    counterpoint fills all registers            → lower ratio.
    try:
        low   = float(Sp[freqs < FREQ_LOW_MAX].sum())
        high  = float(Sp[freqs > FREQ_MID_MAX].sum())
        total = float(Sp.sum()) + 1e-10
        out['mid_register_gap'] = (low + high) / total
    except Exception:
        out['mid_register_gap'] = 0.5

    return out


def main():
    print('─' * 56)
    print('  extract_features2.py  —  targeted supplement')
    print('─' * 56)

    df = pd.read_csv(DATA_CSV)
    print(f'\n  {len(df):,} chunks  |  {df["source_file"].nunique()} source files')

    # Preserve original CSV row order
    ordered_stems = list(dict.fromkeys(df['source_file'].tolist()))

    rows   = []
    errors = 0

    for stem in tqdm(ordered_stems, desc='Files'):
        group    = df[df['source_file'] == stem]
        composer = group['composer'].iloc[0]
        n_chunks = len(group)

        for chunk_idx in range(n_chunks):
            npy_name = f'{stem}_chunk_{chunk_idx:03d}.npy'
            npy_path = os.path.join(PROCESSED_DIR, composer, npy_name)

            try:
                S    = np.load(npy_path)
                feats = extract_new_features(S)
            except Exception as e:
                tqdm.write(f'  WARN: {npy_name}: {e}')
                feats  = {
                    'dynamic_arc_variance': np.nan,
                    'chromatic_saturation': np.nan,
                    'mid_register_gap':     np.nan,
                }
                errors += 1

            rows.append(feats)

    df2 = pd.DataFrame(rows)

    print(f'\n  features.csv  rows : {len(df):,}')
    print(f'  features2.csv rows : {len(df2):,}')
    if len(df2) == len(df):
        print('  Row counts match ✓')
    else:
        print('  ✗ ROW COUNT MISMATCH — something went wrong')

    df2.to_csv(OUT_CSV, index=False)
    print(f'\n  Saved → {OUT_CSV}')
    print('  New columns: dynamic_arc_variance, chromatic_saturation, mid_register_gap')
    if errors:
        print(f'  Chunks with errors (filled NaN): {errors}')
    else:
        print('  No errors ✓')


if __name__ == '__main__':
    main()