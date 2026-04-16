#!/usr/bin/env python3
"""
train.py  —  Phase 3: Hyperparameter Tuning
Systematic RandomizedSearchCV over GradientBoosting params,
then retrains final model with best params + Bach weight.
All Phase 2 diagnostics preserved.
"""

import os
import time
import textwrap
import warnings
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

warnings.filterwarnings('ignore')

# ─── paths ────────────────────────────────────────────────────
DATA_CSV      = os.path.expanduser('~/auralens/data/features.csv')
FEATURES2_CSV = os.path.expanduser('~/auralens/data/features2.csv')
MODEL_DIR     = os.path.expanduser('~/auralens/models')
os.makedirs(MODEL_DIR, exist_ok=True)

COMPOSERS = ['bach', 'vivaldi', 'paganini', 'tchaikovsky']

FAMILY_LABELS = {
    'mfcc':           'MFCC / Timbre            [acoustic]',
    'bv_targeted':    'Bach–Vivaldi targeted    [stylistic]',
    'tv_targeted':    'Tchai–Viv targeted       [stylistic]',
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

# ─── Phase 2 baseline (for final comparison) ──────────────────
PHASE2 = {
    'bach':        0.859,
    'vivaldi':     0.795,
    'paganini':    1.000,
    'tchaikovsky': 0.654,
    'overall':     0.779,
    'bv_confusion': 0.114,
}

# ─── hyperparameter search config ─────────────────────────────
PARAM_DIST = {
    'gb__n_estimators':     randint(150, 700),
    'gb__learning_rate':    uniform(0.02, 0.13),
    'gb__max_depth':        randint(3, 6),
    'gb__subsample':        uniform(0.65, 0.30),
    'gb__min_samples_leaf': randint(1, 10),
    'gb__max_features':     ['sqrt', 'log2'],
}
N_ITER      = 40
N_SPLITS    = 5
BACH_WEIGHT = 2.5

# Phase 2 defaults for comparison display
P2_DEFAULTS = {
    'n_estimators':     350,
    'learning_rate':    0.05,
    'max_depth':        4,
    'subsample':        0.8,
    'min_samples_leaf': 1,
    'max_features':     'sqrt',
}


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

    # ── 3. Hyperparameter search ───────────────────────────────
    section('3. HYPERPARAMETER SEARCH')
    print(f'  Scoring    : balanced_accuracy')
    print(f'  CV         : GroupKFold  (k={N_SPLITS}, piece-level, no leakage)')
    print(f'  Iterations : {N_ITER}')
    print(f'  Parallel   : all CPU cores  (n_jobs=-1)')
    print(f'  This may take 5–15 minutes...\n')

    search_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('gb',     GradientBoostingClassifier(random_state=42)),
    ])

    # Bach weight applied during search too
    search_weights = np.array([BACH_WEIGHT if c == 'bach' else 1.0
                                for c in yc_train])

    gkf = GroupKFold(n_splits=N_SPLITS)
    search = RandomizedSearchCV(
        search_pipe,
        PARAM_DIST,
        n_iter=N_ITER,
        cv=gkf.split(X_train, yc_train, g_train),
        scoring='balanced_accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=False,
    )

    t0 = time.time()
    search.fit(X_train, yc_train,
               gb__sample_weight=search_weights)
    elapsed = time.time() - t0

    print(f'\n  Search complete in {elapsed/60:.1f} minutes')
    print(f'  Best CV balanced_accuracy : {search.best_score_:.4f}')

    # Strip 'gb__' prefix to get clean param dict
    best_params = {k.replace('gb__', ''): v
                   for k, v in search.best_params_.items()}

    print(f'\n  {"Parameter":<22} {"Phase 2":>10} {"Best found":>12}')
    print(f'  {"─" * 46}')
    for param, val in best_params.items():
        p2_val = P2_DEFAULTS.get(param, '—')
        changed = '◄' if str(val) != str(p2_val) else ''
        val_str = f'{val:.4f}' if isinstance(val, float) else str(val)
        p2_str  = f'{p2_val:.4f}' if isinstance(p2_val, float) else str(p2_val)
        print(f'  {param:<22} {p2_str:>10} {val_str:>12}  {changed}')

    # ── 3b. Final training with best params ───────────────────
    section('3b. FINAL TRAINING  (best params + Bach weight)')

    composer_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('gb',     GradientBoostingClassifier(random_state=42, **best_params)),
    ])
    era_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf',     RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    print(f'  Training composer model (GradientBoosting, bach_weight={BACH_WEIGHT})...')
    sample_weights = np.array([BACH_WEIGHT if c == 'bach' else 1.0
                                for c in yc_train])
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

    print('  [B] ARE STYLISTIC FEATURES ACTUALLY USED BY THE MODEL?')
    if stylistic_in_top10:
        print(f'      YES ✓ — {len(stylistic_in_top10)} stylistic family/families '
              f'in top 10: {", ".join(stylistic_in_top10)}')
    elif stylistic_in_top20:
        print(f'      PARTIAL — appear in top 20 but not top 10.')
        print(f'      Families: {", ".join(stylistic_in_top20)}')
    else:
        print('      NO ✗ — no stylistic feature in top 20.')
    print()

    print('  [C] IS BACH–VIVALDI CONFUSION REDUCED?')
    if confusion_rate < 0.15:
        print(f'      YES ✓ — confusion rate {confusion_rate:.1%} is low.')
        print('      Stylistic features are separating the Baroque composers.')
    elif confusion_rate < 0.35:
        print(f'      PARTIAL — confusion rate {confusion_rate:.1%}.')
        print('      Some stylistic separation achieved but overlap remains.')
    else:
        print(f'      NO ✗ — confusion rate {confusion_rate:.1%} is still high.')
    print()

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

    print('  [E] CONCLUSION & NEXT STEP:')
    all_pass = bach_improved and bv_reduced and stylistic_used
    if all_pass and comp_acc >= 0.80:
        conclusion = (
            'Phase 3 tuning successful. All checkpoints pass and overall '
            'accuracy exceeds 80%. Model is ready for deployment or '
            'inference script development.'
        )
    elif all_pass and comp_acc >= 0.75:
        conclusion = (
            'Phase 3 complete. Hyperparameter search improved over Phase 2 '
            'baseline. All validation checkpoints pass. Consider adding more '
            'training data for Tchaikovsky to push chunk-level accuracy further.'
        )
    elif stylistic_used and comp_acc >= 0.60:
        conclusion = (
            'Tuning applied but accuracy gains are modest. The feature set '
            'ceiling has likely been reached. More training data or longer '
            'chunk durations would be needed for further improvement.'
        )
    else:
        conclusion = (
            'Tuning did not improve over Phase 2. Check the best_params '
            'table — if most params are unchanged, the search space may '
            'need widening or more iterations.'
        )
    print(f'      {textwrap.fill(conclusion, width=66, subsequent_indent="      ")}')
    print()

    # ── 13. Phase 2 vs Phase 3 comparison ────────────────────
    section('12. PHASE 2 vs PHASE 3 COMPARISON')

    print(f'\n  {"Metric":<20} {"Phase 2":>9} {"Phase 3":>9} {"Δ":>8}')
    print(f'  {"─" * 50}')

    for comp in COMPOSERS:
        p2  = PHASE2.get(comp, 0)
        p3  = per_comp_acc.get(comp, 0)
        d   = p3 - p2
        sym = '▲' if d > 0.005 else ('▼' if d < -0.005 else '─')
        print(f'  {comp:<20} {p2:>8.1%} {p3:>9.1%} {d:>+7.1%}  {sym}')

    print(f'  {"─" * 50}')

    p2_overall = PHASE2['overall']
    d_overall  = comp_acc - p2_overall
    sym = '▲' if d_overall > 0.005 else ('▼' if d_overall < -0.005 else '─')
    print(f'  {"Overall accuracy":<20} {p2_overall:>8.1%} {comp_acc:>9.1%} '
          f'{d_overall:>+7.1%}  {sym}')

    p2_bv = PHASE2['bv_confusion']
    d_bv  = confusion_rate - p2_bv
    sym = '▲' if d_bv > 0.005 else ('▼' if d_bv < -0.005 else '─')
    print(f'  {"B–V confusion":<20} {p2_bv:>8.1%} {confusion_rate:>9.1%} '
          f'{d_bv:>+7.1%}  {sym}')

    print()


if __name__ == '__main__':
    main()