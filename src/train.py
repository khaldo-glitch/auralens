import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     ConfusionMatrixDisplay)

# ─── paths ───────────────────────────────────────────────────
BASE_DIR     = os.path.expanduser("~/auralens")
FEATURES_CSV = os.path.join(BASE_DIR, "data", "features.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "data", "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ─── settings ────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
N_ESTIMATORS   = 500
TOP_N_FEATURES = 20

NON_FEATURE_COLS = ['composer', 'era', 'key_mode', 'source_file']
# ─────────────────────────────────────────────────────────────


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    print(f"loaded {len(df)} rows x {len(df.columns)} columns")

    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        print(f"WARNING: dropped {n_before - len(df)} rows with NaN values")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X            = df[feature_cols].values
    y_composer   = df['composer'].values
    y_era        = df['era'].values
    source_files = df['source_file'].values

    print(f"feature matrix: {X.shape[0]} chunks x {X.shape[1]} features")

    print("\ncomposer distribution:")
    labels, counts = np.unique(y_composer, return_counts=True)
    for label, count in zip(labels, counts):
        bar = 'x' * (count // 50)
        print(f"  {label:<14} {count:>5}  {bar}")

    print("\nera distribution:")
    labels, counts = np.unique(y_era, return_counts=True)
    for label, count in zip(labels, counts):
        bar = 'x' * (count // 100)
        print(f"  {label:<14} {count:>5}  {bar}")

    return X, y_composer, y_era, feature_cols, source_files


def build_composer_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(
            n_estimators  = 300,
            max_depth     = 5,
            learning_rate = 0.1,
            random_state  = RANDOM_STATE
        ))
    ])


def build_era_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators = N_ESTIMATORS,
            class_weight = 'balanced',
            random_state = RANDOM_STATE,
            n_jobs       = -1
        ))
    ])


def evaluate(model, X_test, y_test, label, class_names):
    print(f"\n{'=' * 52}")
    print(f"  {label.upper()} MODEL — TEST RESULTS")
    print(f"{'=' * 52}")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{label} classifier — confusion matrix")
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"confusion_{label}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"confusion matrix saved -> {plot_path}")

    return y_pred


def plot_feature_importances(model, feature_cols, label):
    # get the classifier step — either 'gb' or 'rf'
    clf = model.named_steps.get('gb') or model.named_steps.get('rf')
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1][:TOP_N_FEATURES]

    top_names  = [feature_cols[i] for i in indices]
    top_scores = importances[indices]

    print(f"\ntop {TOP_N_FEATURES} features ({label}):")
    for name, score in zip(top_names, top_scores):
        bar = 'x' * int(score * 500)
        print(f"  {name:<35} {score:.4f}  {bar}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(TOP_N_FEATURES), top_scores[::-1], color='steelblue')
    ax.set_yticks(range(TOP_N_FEATURES))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("feature importance (mean decrease in impurity)")
    ax.set_title(f"{label} classifier — top {TOP_N_FEATURES} features")
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"importances_{label}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"feature importances saved -> {plot_path}")


def train_model(X, y, label, feature_cols, groups, pipeline_fn):
    print(f"\n{'-' * 52}")
    print(f"  training {label} model")
    print(f"{'-' * 52}")

    class_names = sorted(set(y))

    gss = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"train set: {len(X_train)} chunks  "
          f"({len(set(groups[train_idx]))} pieces)")
    print(f"test  set: {len(X_test)} chunks  "
          f"({len(set(groups[test_idx]))} pieces)")

    model = pipeline_fn()
    print(f"training... ", end='', flush=True)
    model.fit(X_train, y_train)
    print("done")

    evaluate(model, X_test, y_test, label, class_names)
    plot_feature_importances(model, feature_cols, label)

    model_path = os.path.join(MODELS_DIR, f"{label}_model.pkl")
    joblib.dump(model, model_path)
    print(f"model saved -> {model_path}")

    return model


def main():
    print("=" * 52)
    print("  AURALENS — model training")
    print("=" * 52)

    X, y_composer, y_era, feature_cols, source_files = load_data()

    composer_model = train_model(
        X, y_composer, "composer", feature_cols, source_files,
        build_composer_pipeline)

    era_model = train_model(
        X, y_era, "era", feature_cols, source_files,
        build_era_pipeline)

    print(f"\n{'=' * 52}")
    print("  all done.")
    print(f"  models saved in: {MODELS_DIR}")
    print(f"  plots  saved in: {PLOTS_DIR}")
    print(f"{'=' * 52}")


if __name__ == "__main__":
    main()
