import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

CSV_PATH = os.path.expanduser("~/auralens/data/features.csv")
OUT_DIR  = os.path.expanduser("~/auralens/data/plots")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "bach":        "#534AB7",
    "vivaldi":     "#1D9E75",
    "paganini":    "#D85A30",
    "tchaikovsky": "#BA7517"
}

df = pd.read_csv(CSV_PATH)
composers    = ["bach", "vivaldi", "paganini", "tchaikovsky"]
feature_cols = [c for c in df.columns if c not in ("composer", "era")]

print(f"Loaded {len(df)} rows x {len(feature_cols)} features")
print(f"Composer counts:\n{df['composer'].value_counts()}\n")

# ── Plot 1: Composer distribution (sanity check) ─────────────
fig, ax = plt.subplots(figsize=(8, 4))
counts  = df['composer'].value_counts().reindex(composers)
bars    = ax.bar(composers, counts.values,
                 color=[COLORS[c] for c in composers])
ax.set_title("Number of 30-second chunks per composer")
ax.set_ylabel("chunks")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(val), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_chunk_counts.png"), dpi=150)
plt.close()
print("saved 01_chunk_counts.png")

# ── Plot 2: Box plots for most discriminative features ────────
key_features = [
    "tempo", "onset_rate", "rhythm_regularity",
    "dynamic_range", "rms_std",
    "key_clarity", "key_mode", "chromaticism", "tonal_instability",
    "chroma_entropy", "hp_ratio", "syncopation",
    "self_similarity_peakiness", "polyphony_estimate"
]
key_features = [f for f in key_features if f in df.columns]

n_cols = 3
n_rows = (len(key_features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(16, n_rows * 3.5))
axes = axes.flatten()

for idx, feat in enumerate(key_features):
    ax = axes[idx]
    data = [df[df['composer'] == c][feat].dropna().values
            for c in composers]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2))
    for patch, c in zip(bp['boxes'], composers):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.8)
    ax.set_title(feat, fontsize=10)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["B", "V", "P", "T"], fontsize=9)
    ax.grid(axis='y', alpha=0.3)

for idx in range(len(key_features), len(axes)):
    axes[idx].set_visible(False)

legend_patches = [mpatches.Patch(color=COLORS[c], label=c)
                  for c in composers]
fig.legend(handles=legend_patches, loc='lower right',
           fontsize=10, ncol=2)
fig.suptitle("Feature distributions per composer  (B=Bach  V=Vivaldi  P=Paganini  T=Tchaikovsky)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_key_features_boxplot.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("saved 02_key_features_boxplot.png")

# ── Plot 3: Chroma mean per composer ─────────────────────────
pitch_classes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
chroma_cols   = [f'chroma_mean_{i}' for i in range(12)]
chroma_cols   = [c for c in chroma_cols if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for idx, composer in enumerate(composers):
    ax      = axes[idx]
    subset  = df[df['composer'] == composer][chroma_cols]
    means   = subset.mean().values
    means   = means / (means.sum() + 1e-10)
    ax.bar(pitch_classes, means, color=COLORS[composer], alpha=0.85)
    ax.set_title(composer.capitalize(), fontsize=12)
    ax.set_ylabel("relative energy")
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle("Average chroma profile per composer\n(which pitch classes dominate)",
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_chroma_profiles.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("saved 03_chroma_profiles.png")

# ── Plot 4: Tempo distribution ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for composer in composers:
    vals = df[df['composer'] == composer]['tempo'].dropna()
    ax.hist(vals, bins=40, alpha=0.55,
            color=COLORS[composer], label=composer, density=True)
ax.set_title("Tempo distribution per composer")
ax.set_xlabel("BPM")
ax.set_ylabel("density")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_tempo_distribution.png"),
            dpi=150)
plt.close()
print("saved 04_tempo_distribution.png")

# ── Plot 5: Correlation heatmap of key features ───────────────
corr_data = df[key_features + ['composer']].copy()
corr_data['composer_code'] = pd.Categorical(
    corr_data['composer']).codes
corr_matrix = corr_data.drop(columns='composer').corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r',
               vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)
labels = list(corr_matrix.columns)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_title("Feature correlation matrix", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_correlation_heatmap.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("saved 05_correlation_heatmap.png")

# ── Plot 6: PCA — do composers cluster? ───────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X      = df[feature_cols].fillna(0).values
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X_sc)

fig, ax = plt.subplots(figsize=(10, 8))
for composer in composers:
    mask = df['composer'].values == composer
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLORS[composer], label=composer,
               alpha=0.3, s=8)

ax.set_title(
    f"PCA of all features  —  var explained: "
    f"{pca.explained_variance_ratio_[0]*100:.1f}% + "
    f"{pca.explained_variance_ratio_[1]*100:.1f}%",
    fontsize=11)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(markerscale=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_pca_scatter.png"),
            dpi=150)
plt.close()
print("saved 06_pca_scatter.png")

print(f"\nAll plots saved to {OUT_DIR}")

# ── Plot 7: MFCC means per composer ──────────────────────────
mfcc_mean_cols = [f'mfcc_mean_{i}' for i in range(13)]
mfcc_mean_cols = [c for c in mfcc_mean_cols if c in df.columns]

if mfcc_mean_cols:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(13)
    width = 0.2
    for idx, composer in enumerate(composers):
        means = df[df['composer'] == composer][mfcc_mean_cols].mean().values
        ax.bar(x + idx * width, means, width,
               label=composer, color=COLORS[composer], alpha=0.85)
    ax.set_title("MFCC mean values per composer")
    ax.set_xlabel("MFCC coefficient index")
    ax.set_ylabel("mean value")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'MFCC {i}' for i in range(13)], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_mfcc_means.png"), dpi=150)
    plt.close()
    print("saved 07_mfcc_means.png")

# ── Plot 8: MFCC box plots ────────────────────────────────────
if mfcc_mean_cols:
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()
    for i in range(13):
        ax   = axes[i]
        feat = f'mfcc_mean_{i}'
        if feat not in df.columns:
            continue
        data = [df[df['composer'] == c][feat].dropna().values
                for c in composers]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        for patch, c in zip(bp['boxes'], composers):
            patch.set_facecolor(COLORS[c])
            patch.set_alpha(0.8)
        ax.set_title(f'MFCC {i}', fontsize=10)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(['B', 'V', 'P', 'T'], fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    for i in range(13, len(axes)):
        axes[i].set_visible(False)

    legend_patches = [mpatches.Patch(color=COLORS[c], label=c)
                      for c in composers]
    fig.legend(handles=legend_patches, loc='lower right', fontsize=10)
    fig.suptitle("MFCC distributions per composer", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_mfcc_boxplots.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("saved 08_mfcc_boxplots.png")