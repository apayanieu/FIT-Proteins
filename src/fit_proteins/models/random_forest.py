"""
This script trains, evaluates, and finalizes a Random Forest classifier for BRD4 binding
prediction using the official block-aware train/validation split defined in
`train_brd4_50k_clean_blocks.parquet`.

The block-aware split divides the 50k compounds into:
    • train_in  – molecules whose building blocks appear in the training chemistry domain
    • val_ood   – molecules containing at least one building block NOT seen during training

This out-of-distribution (OOD) validation set is crucial because it measures whether the model
can generalize to novel chemical fragments, as required by the BRD4 project design.

The script performs the following steps:
1. Loads ECFP fingerprint matrices, activity labels, molecule IDs, and the official split metadata.
2. Separates the training data into train_in (used for fitting) and val_ood (used only for testing).
3. Trains a baseline Random Forest model on train_in and evaluates it on val_ood using:
       – Average Precision (AP): the competition's primary metric
       – ROC-AUC: class separability on unseen chemistries
4. After evaluation, the model is retrained on the full labeled dataset
   (train_in + val_ood) to maximize learning before generating test-set predictions.

This workflow strictly follows the project requirements:
    – evaluate on chemically novel OOD examples
    – prevent leakage across building blocks
    – retrain on all available training data before inference
ensuring that the final Random Forest model is ready for BRD4 test-set prediction and Kaggle submission.
"""

# ============================================================
# RANDOM FOREST WITH BLOCK-AWARE SPLIT AND FINAL FULL TRAINING
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------------------------
# ENSURE DATA FILES EXIST
# -------------------------------------------
DATA_DIR = Path("../../../data/processed")

def ensure_exists(path):
    """Raises a clear error if a required file does not exist."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required file not found: {path!s}."
            "Place files under data/processed or correct the path."
        )

# File paths
X_train_full_fp = DATA_DIR / "X_train_full.npz"
X_test_fp       = DATA_DIR / "X_test.npz"
y_train_fp      = DATA_DIR / "y_train_full.npy"
ids_train_fp    = DATA_DIR / "ids_train_full.npy"
ids_test_fp     = DATA_DIR / "ids_test.npy"
splits_fp       = DATA_DIR / "train_brd4_50k_clean_blocks.parquet"

for p in (X_train_full_fp, X_test_fp, y_train_fp, ids_train_fp, ids_test_fp, splits_fp):
    ensure_exists(p)

# -------------------------------------------
# LOAD DATA & SPLIT METADATA
# -------------------------------------------
X_train_full = load_npz(X_train_full_fp)
X_test       = load_npz(X_test_fp)

y_train_full   = np.load(y_train_fp)
ids_train_full = np.load(ids_train_fp)
ids_test       = np.load(ids_test_fp)

splits_df = pd.read_parquet(splits_fp)

print("X_train_full shape:", X_train_full.shape)
print("X_test shape:", X_test.shape)
print("y_train_full shape:", y_train_full.shape)
print("ids_train_full shape:", ids_train_full.shape)
print("ids_test shape:", ids_test.shape)
print("Splits columns:", splits_df.columns.tolist())

# -------------------------------------------
# APPLY THE OFFICIAL BLOCK-AWARE SPLIT
# -------------------------------------------
df = pd.DataFrame({"id": ids_train_full})
df = df.merge(splits_df[["id", "split_group"]], on="id", how="left")

train_mask = df["split_group"] == "train_in"
val_mask   = df["split_group"] == "val_ood"

# train_in split → used for initial training
X_tr = X_train_full[train_mask.values]
y_tr = y_train_full[train_mask.values]

# val_ood split → used ONLY for evaluation (never for training)
X_val = X_train_full[val_mask.values]
y_val = y_train_full[val_mask.values]

print("Train_in shape:", X_tr.shape, " | Positives:", y_tr.sum())
print("Val_ood shape:", X_val.shape, " | Positives:", y_val.sum())

# -------------------------------------------
# STEP 1 — TRAIN ON train_in (for validation evaluation)
# -------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=250,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

rf_clf.fit(X_tr, y_tr)
print("Random Forest trained on train_in.")

# -------------------------------------------
# STEP 2 — EVALUATE ON val_ood (UNSEEN DATA)
# -------------------------------------------
val_proba_rf = rf_clf.predict_proba(X_val)[:, 1]

ap_val_rf = average_precision_score(y_val, val_proba_rf)
roc_val_rf = roc_auc_score(y_val, val_proba_rf)

print(f"RF – Val_OOD AP:  {ap_val_rf:.6f}")
print(f"RF – Val_OOD AUC: {roc_val_rf:.6f}")

# ---------------------------------------------------------
# STEP 3 — RETRAIN ON *ALL* TRAINING DATA (train_in + val_ood)
# ---------------------------------------------------------
# This model will be used to predict X_test for Kaggle submission.
rf_final = RandomForestClassifier(
    n_estimators=250,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

rf_final.fit(X_train_full, y_train_full)
print("Random Forest retrained on FULL training set.")
