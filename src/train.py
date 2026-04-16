#!/usr/bin/env python3
"""
train.py  —  Phase 2: Diagnosis & Validation
Retrains on the enriched 156-feature set and produces a full diagnostic report.

Constraints:
  - Feature extraction unchanged
  - Model architecture unchanged  (GB=composer, RF=era)
  - No hyperparameter tuning
"""

import os
import textwrap
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

warnings.filterwarnings('ignore')

# ─── paths ────────────────────────────────────────────────────
DATA_CSV  = os.path.expanduser('~/auralens/data/features.csv')
FEATURES2_CSV = os.path.expanduser('~/auralens/data/features2.csv')  
MODEL_DIR = os.path.expanduser('~/auralens/models')
os.makedirs(MODEL_DIR, exist_ok=True)

COMPOSERS = ['bach', 'vivaldi', 'paganini', 'tchaikovsky']

FAMILY_LABELS = {
    'mfcc':           'MFCC / Timbre            [acoustic]',
    'bv_targeted':    'Bach–Vivaldi targeted    [stylistic]',
    'tv_targeted':    'Tchai–Viv targeted      [stylistic]',
    'spectral':       'Spectral shape           [acoustic]',
    'harmony':        'Harmony / Chroma         [acoustic]',
    'rhythm':         'Rhythm / Tempo           [acoustic]',
    'dynamics':       'Dynamics / Energy        [acoustic]',
    'texture':        'Texture (HP ratio)       [acoustic]',
    'structure':      'Self-similarity          [acoustic]',
    'polyphony':      'Polyphony estimate       [acoustic]',
    'horiz_vert':     'Horiz. vs Vert. motion   [stylistic]',
    'register':       'Register behavior        [stylistic]',
    'voice_indep':    'Voice independence       [stylistic]',
    'melodic_motion': 'Melodic motion grammar   [stylistic]',
    'imitation':      'Imitation / Counterpt.   [stylistic]',
    'motivic':        'Motivic development      [stylistic]',
    'phrase':         'Phrase structure         [stylistic]',
    'temporal':       'Temporal organization    [stylistic]',
}

ACOUSTIC_FAM  = {'mfcc', 'spectral', 'harmony', 'rhythm',
                 'dynamics', 'texture', 'structure', 'polyphony'}
STYLISTIC_FAM = {'horiz_vert', 'register', 'voice_indep', 'melodic_motion',
                 'imitation', 'motivic', 'phrase', 'temporal',
                 'bv_targeted', 'tv_targeted'}

# ─── feature family tagger ────────────────────────────────────
def tag_feature(name):
    if name.startswith(('mfcc_', 'delta_', 'delta2_')):
        return 'mfcc'
    if name.startswith('spectral_') and name not in ('spectral_peak_count', 'spectral_flux'):
        return 'spectral'
    if (name.startswith(('chroma_', 'tonnetz_')) or name in
            ('key_clarity', 'key_mode', 'chromaticism', 'tonal_instability')):
        return 'harmony'
    if name in ('tempo', 'onset_rate', 'rhythm_regularity',
                'tempogram_entropy', 'syncopation'):
        return 'rhythm'
    if name in ('rms_mean', 'rms_std', 'dynamic_range',
                'loudness_slope', 'spectral_flux', 'zcr_mean'):
        return 'dynamics'
    if name == 'hp_ratio':
        return 'texture'
    if name in ('self_similarity_peakiness', 'autocorr_peak'):
        return 'structure'
    if name in ('spectral_peak_count', 'polyphony_estimate'):
        return 'polyphony'
    if name in ('horizontal_motion_ratio', 'chordal_jump_density',
                'melodic_continuity'):
        return 'horiz_vert'
    if name.startswith('register_') or name == 'band_activity_balance':
        return 'register'
    if name.startswith('band_corr') or name in ('independence_index',
                                                 'anti_homophony'):
        return 'voice_indep'
    if name in ('stepwise_motion_ratio', 'leap_ratio', 'large_leap_ratio',
                'avg_interval_size', 'direction_balance',
                'motion_variety', 'stasis_ratio'):
        return 'melodic_motion'
    if name in ('chroma_lag_corr_1s', 'chroma_lag_corr_2s',
                'chroma_lag_corr_4s', 'imitation_density',
                'transposition_invariant_recurrence', 'lag_profile_entropy'):
        return 'imitation'
    if name.startswith(('ngram', 'motive_')) or name == 'top_motive_dominance':
        return 'motivic'
    if name in ('phrase_boundary_sharpness', 'continuity_index',
                'phrase_length_regularity', 'cadential_density'):
        return 'phrase'
    if name in ('pitch_motion_autocorr_long', 'tonal_center_stability',
                'activity_periodicity', 'density_trend',
                'long_range_chroma_consistency'):
        return 'temporal'
    if name in ('bass_melodic_complexity', 'harmonic_rhythm_rate',
                'voice_interval_mirroring', 'note_repetition_rate',
                'sequence_periodicity'):
        return 'bv_targeted'
    if name in ('dynamic_arc_variance', 'chromatic_saturation', 'mid_register_gap'):
        return 'tv_targeted'
    return 'other'


# ─── display helpers ──────────────────────────────────────────
def bar(value, width=28, max_val=1.0):
    filled = int(round(value / (max_val + 1e-10) * width))
    filled = max(0, min(filled, width))
    return '█' * filled + '░' * (width - filled)

def section(title):
    print(f'\n{"═" * 62}')
    print(f'  {title}')
    print(f'{"═" * 62}')


# ─── main ─────────────────────────────────────────────────────
def main():

    # ── 1. Load ───────────────────────────────────────────────
    section('1. LOADING DATA')
    df = pd.read_csv(DATA_CSV)
    # Merge supplemental features if available
    if os.path.exists(FEATURES2_CSV):
        df2      = pd.read_csv(FEATURES2_CSV)
        new_cols = [c for c in df2.columns if c != 'source_file']
        if len(df2) == len(df):
            df = pd.concat([df, df2[new_cols]], axis=1)
            print(f'  Merged features2.csv  (+{len(new_cols)} features)')
        else:
            print(f'  WARNING: features2.csv row count mismatch — skipped')
    print(f'  Loaded {len(df)} chunks')

    drop_cols    = {'composer', 'era', 'source_file'}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    families     = {c: tag_feature(c) for c in feature_cols}

    acoustic_feats  = [c for c in feature_cols if families[c] in ACOUSTIC_FAM]
    stylistic_feats = [c for c in feature_cols if families[c] in STYLISTIC_FAM]

    print(f'  Total features  : {len(feature_cols)}')
    print(f'  Acoustic        : {len(acoustic_feats)}')
    print(f'  Stylistic (new) : {len(stylistic_feats)}')

    # ── 2. Split ──────────────────────────────────────────────
    section('2. PIECE-LEVEL TRAIN / TEST SPLIT')
    X      = df[feature_cols].values
    y_comp = df['composer'].values
    y_era  = df['era'].values
    groups = df['source_file'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_comp, groups=groups))

    X_train, X_test   = X[train_idx],      X[test_idx]
    yc_train, yc_test = y_comp[train_idx], y_comp[test_idx]
    ye_train, ye_test = y_era[train_idx],  y_era[test_idx]
    g_train, g_test   = groups[train_idx], groups[test_idx]

    print(f'  Train: {len(X_train)} chunks, {len(set(g_train))} pieces')
    print(f'  Test : {len(X_test)} chunks, {len(set(g_test))} pieces')
    print(f'\n  Test-set composer distribution:')
    for comp in COMPOSERS:
        n = int(np.sum(yc_test == comp))
        print(f'    {comp:<14} {n:>4} chunks')

    # ── 3. Train ──────────────────────────────────────────────
    section('3. TRAINING  (architecture unchanged)')

    composer_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('gb',     GradientBoostingClassifier(
            random_state=42,
            max_features='sqrt',
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
        ))
    ])
    era_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf',     RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print('  Training composer model (GradientBoosting)...')
    weight_map = {
        'bach':        2.5,
        'vivaldi':     1.0,
        'paganini':    1.0,
        'tchaikovsky': 1.0,
    } 
    sample_weights = np.array([weight_map[c] for c in yc_train])
    composer_pipe.fit(X_train, yc_train, gb__sample_weight=sample_weights)

    print('  Training RandomForest      → era      ...')
    era_pipe.fit(X_train, ye_train)

    joblib.dump(composer_pipe, os.path.join(MODEL_DIR, 'composer_model.pkl'))
    joblib.dump(era_pipe,      os.path.join(MODEL_DIR, 'era_model.pkl'))
    print('  Models saved.')

    # ── 4. Predictions ────────────────────────────────────────
    yc_pred = composer_pipe.predict(X_test)
    ye_pred = era_pipe.predict(X_test)

    # ── 5. Accuracy ───────────────────────────────────────────
    section('4. OVERALL ACCURACY')

    comp_acc = accuracy_score(yc_test, yc_pred)
    era_acc  = accuracy_score(ye_test, ye_pred)
    print(f'\n  Composer  {bar(comp_acc)}  {comp_acc:.1%}  (baseline 25%)')
    print(f'  Era       {bar(era_acc)}  {era_acc:.1%}  (baseline 50%)')

        # ── 6. Per-composer accuracy ──────────────────────────────
    section('5. PER-COMPOSER ACCURACY')

    print()
    per_comp_acc = {}
    for comp in COMPOSERS:
        mask = yc_test == comp
        n    = int(mask.sum())
        if n == 0:
            per_comp_acc[comp] = 0.0
            print(f'  {comp:<14}  — no test samples')
            continue
        acc = accuracy_score(yc_test[mask], yc_pred[mask])
        per_comp_acc[comp] = acc
        print(f'  {comp:<14} {bar(acc)}  {acc:.1%}  ({n} chunks)')

    # ── 6b. Piece-level aggregation ───────────────────────────
    section('5b. PIECE-LEVEL EVALUATION')

    proba   = composer_pipe.predict_proba(X_test)
    classes = list(composer_pipe.classes_)

    piece_true  = {}
    piece_proba = {}
    for piece, true_comp, chunk_p in zip(g_test, yc_test, proba):
        piece_true[piece] = true_comp
        if piece not in piece_proba:
            piece_proba[piece] = []
        piece_proba[piece].append(chunk_p)

    pieces  = list(piece_true.keys())
    yp_true = [piece_true[p] for p in pieces]
    yp_pred = [classes[np.argmax(np.mean(piece_proba[p], axis=0))]
               for p in pieces]

    piece_acc = accuracy_score(yp_true, yp_pred)
    print(f'\n  Chunk-level accuracy : {accuracy_score(yc_test, yc_pred):.1%}')
    print(f'  Piece-level accuracy : {piece_acc:.1%}  ({len(pieces)} pieces)')
    print(f'  Aggregation lift     : {piece_acc - accuracy_score(yc_test, yc_pred):+.1%}')

    print(f'\n  Per-composer (piece-level):')
    for comp in COMPOSERS:
        mask = [t == comp for t in yp_true]
        n    = sum(mask)
        if n == 0:
            continue
        t   = [yp_true[i] for i, m in enumerate(mask) if m]
        p   = [yp_pred[i] for i, m in enumerate(mask) if m]
        acc = accuracy_score(t, p)
        print(f'  {comp:<14} {bar(acc)}  {acc:.1%}  [{n} pieces]')

    # ── 7. Confusion matrix ───────────────────────────────────
    section('6. CONFUSION MATRIX  (row = true, col = predicted)')
    cm = confusion_matrix(yc_test, yc_pred, labels=COMPOSERS)
    short = [c[:7] for c in COMPOSERS]
    header = f'  {"":>14}' + ''.join(f'{s:>9}' for s in short)
    print(f'\n{header}')
    print(f'  {"─" * (14 + 9 * len(COMPOSERS))}')
    for i, comp in enumerate(COMPOSERS):
        row_str = ''.join(f'{cm[i,j]:>9}' for j in range(len(COMPOSERS)))
        acc_str = f'{per_comp_acc[comp]:.0%}'
        print(f'  {comp:<14}{row_str}    ← {acc_str}')

    # ── 8. Feature importance — top 30 ───────────────────────
    section('7. TOP 30 FEATURES  (composer model)')

    importances = composer_pipe.named_steps['gb'].feature_importances_
    fi_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': importances,
        'family':     [families[c] for c in feature_cols],
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f'\n  {"#":<4} {"Feature":<38} {"Family":<18} {"Imp":>8}')
    print(f'  {"─" * 74}')
    for i, r in fi_df.head(30).iterrows():
        is_new = '★' if r['family'] in STYLISTIC_FAM else ' '
        print(f'  {i+1:<4} {r["feature"]:<38} {r["family"]:<18} '
              f'{r["importance"]:>8.4f} {is_new}')
    print(f'\n  ★ = stylistic (new) feature')

    # ── 9. Feature importance — by family ────────────────────
    section('8. IMPORTANCE BY FEATURE FAMILY')

    family_imp = (fi_df.groupby('family')['importance']
                       .sum()
                       .sort_values(ascending=False))
    total_imp = family_imp.sum()
    max_imp   = float(family_imp.max())

    print()
    for fam, imp in family_imp.items():
        label = FAMILY_LABELS.get(fam, fam)
        b     = bar(imp, width=22, max_val=max_imp)
        pct   = imp / total_imp * 100
        print(f'  {label:<43} {b}  {pct:5.1f}%')

    # ── 10. Bach–Vivaldi confusion detail ────────────────────
    section('9. BACH–VIVALDI CONFUSION DETAIL')

    b_as_v  = int(np.sum((yc_test == 'bach')    & (yc_pred == 'vivaldi')))
    v_as_b  = int(np.sum((yc_test == 'vivaldi') & (yc_pred == 'bach')))
    b_total = int(np.sum(yc_test == 'bach'))
    v_total = int(np.sum(yc_test == 'vivaldi'))

    print(f'\n  Bach    → predicted Vivaldi : {b_as_v:>4} / {b_total}  '
          f'({b_as_v/(b_total+1e-10):.1%})')
    print(f'  Vivaldi → predicted Bach    : {v_as_b:>4} / {v_total}  '
          f'({v_as_b/(v_total+1e-10):.1%})')

    confusion_rate = (b_as_v + v_as_b) / max(b_total + v_total, 1)
    print(f'\n  Combined confusion rate     : {confusion_rate:.1%}')

    # mean stylistic feature comparison Bach vs Vivaldi
    df_test = df.iloc[test_idx].copy()
    key_stylistic = [
        'independence_index', 'imitation_density',
        'horizontal_motion_ratio', 'band_activity_balance',
        'ngram3_repetition_ratio', 'register_crossing_rate',
    ]
    key_stylistic = [f for f in key_stylistic if f in df_test.columns]

    if key_stylistic:
        print(f'\n  Key stylistic features — Bach vs Vivaldi mean values:')
        print(f'\n  {"Feature":<35} {"Bach":>8} {"Vivaldi":>9} {"Δ":>8}')
        print(f'  {"─" * 62}')
        for feat in key_stylistic:
            b_mean = float(df_test.loc[df_test['composer']=='bach',    feat].mean())
            v_mean = float(df_test.loc[df_test['composer']=='vivaldi', feat].mean())
            delta  = b_mean - v_mean
            flag   = '◄' if abs(delta) > 0.05 else ''
            print(f'  {feat:<35} {b_mean:>8.3f} {v_mean:>9.3f} {delta:>+8.3f} {flag}')

    # ── 11. Validation checkpoints ────────────────────────────
    section('10. VALIDATION CHECKPOINTS')

    top20_fam = set(fi_df.head(20)['family'].tolist())
    top10_fam = set(fi_df.head(10)['family'].tolist())
    stylistic_in_top20 = [f for f in top20_fam if f in STYLISTIC_FAM]
    stylistic_in_top10 = [f for f in top10_fam if f in STYLISTIC_FAM]

    stylistic_imp_total = fi_df[fi_df['family'].isin(STYLISTIC_FAM)]['importance'].sum()

    def check(ok):
        return '✓' if ok else '✗'

    bach_improved   = per_comp_acc.get('bach', 0) > 0.35
    bv_reduced      = confusion_rate < 0.35
    stylistic_used  = len(stylistic_in_top20) > 0
    stylistic_share = stylistic_imp_total / total_imp

    print(f'\n  {check(bach_improved)}  Bach accuracy > 35%          '
          f'{per_comp_acc.get("bach",0):.1%}')
    print(f'  {check(bv_reduced)}  Bach–Vivaldi confusion < 35% '
          f'{confusion_rate:.1%}')
    print(f'  {check(stylistic_used)}  Stylistic features in top 20 '
          f'{len(stylistic_in_top20)} families')
    print(f'  {check(stylistic_share > 0.10)}  Stylistic share > 10%        '
          f'{stylistic_share:.1%}')

    # ── 12. Written diagnosis ─────────────────────────────────
    section('11. WRITTEN DIAGNOSIS')
    print()

    # --- [A] Feature presence ---
    print('  [A] ARE STYLISTIC FEATURES PRESENT IN FEATURE SPACE?')
    print(f'      {len(stylistic_feats)} stylistic features extracted '
          f'({stylistic_share*100:.1f}% of model importance).')
    if stylistic_share > 0.15:
        print('      Assessment: YES ✓ — substantial contribution.')
    elif stylistic_share > 0.05:
        print('      Assessment: MARGINAL — present but acoustics dominate.')
    else:
        print('      Assessment: NO ✗ — stylistic features carry almost no signal.')
    print()

    # --- [B] Feature usage ---
    print('  [B] ARE STYLISTIC FEATURES ACTUALLY USED BY THE MODEL?')
    if stylistic_in_top10:
        print(f'      YES ✓ — {len(stylistic_in_top10)} stylistic family/families '
              f'in top 10: {", ".join(stylistic_in_top10)}')
    elif stylistic_in_top20:
        print(f'      PARTIAL — appear in top 20 but not top 10.')
        print(f'      Families: {", ".join(stylistic_in_top20)}')
        print(f'      MFCCs and spectral features still dominate top decisions.')
    else:
        print('      NO ✗ — no stylistic feature in top 20.')
        print('      The model relies entirely on acoustic timbre.')
    print()

    # --- [C] Bach–Vivaldi ---
    print('  [C] IS BACH–VIVALDI CONFUSION REDUCED?')
    if confusion_rate < 0.15:
        print(f'      YES ✓ — confusion rate {confusion_rate:.1%} is low.')
        print('      Stylistic features are separating the Baroque composers.')
    elif confusion_rate < 0.35:
        print(f'      PARTIAL — confusion rate {confusion_rate:.1%}.')
        print('      Some stylistic separation achieved but overlap remains.')
        print('      This is expected: both composers wrote for similar instruments.')
    else:
        print(f'      NO ✗ — confusion rate {confusion_rate:.1%} is still high.')
        print('      Baroque composers remain acoustically and stylistically similar')
        print('      within 30-second chunks. Longer context may be required.')
    print()

    # --- [D] Per-composer ---
    print('  [D] PER-COMPOSER ASSESSMENT:')
    for comp in COMPOSERS:
        acc = per_comp_acc.get(comp, 0)
        if acc >= 0.75:
            note = 'well-separated, distinctive features dominate'
        elif acc >= 0.50:
            note = 'moderate — partial confusion with similar composers'
        else:
            note = 'poor — representation gap or class imbalance issue'
        print(f'      {comp:<14} {acc:.0%}  — {note}')
    print()

    # --- [E] Conclusion ---
    print('  [E] CONCLUSION & NEXT STEP:')
    all_pass = bach_improved and bv_reduced and stylistic_used
    if all_pass and comp_acc >= 0.75:
        conclusion = (
            'Representation is validated. Stylistic features contribute '
            'meaningfully, Bach accuracy improved, and Baroque confusion is '
            'reduced. Safe to proceed to Phase 3 (hyperparameter tuning).'
        )
    elif stylistic_used and comp_acc >= 0.60:
        conclusion = (
            'Representation is partially validated. Stylistic features are '
            'present and used, accuracy is reasonable. Proceed to Phase 3 '
            '(hyperparameter tuning) rather than adding more features.'
        )
    elif stylistic_used:
        conclusion = (
            'Stylistic features are used but accuracy is still low. Remaining '
            'errors likely reflect a representational gap — the current '
            'features do not fully resolve Baroque ambiguity at 30-second '
            'granularity. Do NOT proceed to tuning yet. Consider longer '
            'chunks or ensemble aggregation across chunks.'
        )
    else:
        conclusion = (
            'Stylistic features are not being used. The model falls back '
            'entirely on acoustic timbre. Either the new features carry '
            'no discriminative signal at 30-second chunk level, or the '
            'chroma-based proxies (imitation, voice independence) are too '
            'noisy to generalise. Revisit feature design before Phase 3.'
        )
    print(f'      {textwrap.fill(conclusion, width=66, subsequent_indent="      ")}')
    print()


if __name__ == '__main__':
    main()