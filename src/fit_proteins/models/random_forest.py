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
# RANDOM FOREST USING PRE-SPLIT ECFP DATA (TRAIN/VAL_OOD/TEST_OOD)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz, vstack  # vstack to combine sparse matrices

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
            " Place files under data/processed or correct the path."
        )

# Pre-split ECFP files (all in the same folder as this script expects)
X_train_in_fp   = DATA_DIR / "X_train_in_ecfp.npz"
y_train_in_fp   = DATA_DIR / "y_train_in.npy"

X_val_ood_fp    = DATA_DIR / "X_val_ood_ecfp.npz"
y_val_ood_fp    = DATA_DIR / "y_val_ood.npy"

X_test_ood_fp   = DATA_DIR / "X_test_ood_ecfp.npz"
y_test_ood_fp   = DATA_DIR / "y_test_ood.npy"

for p in (X_train_in_fp, y_train_in_fp,
          X_val_ood_fp,  y_val_ood_fp,
          X_test_ood_fp, y_test_ood_fp):
    ensure_exists(p)

# -------------------------------------------
# LOAD PRE-SPLIT DATA
# -------------------------------------------
X_tr   = load_npz(X_train_in_fp)     # train_in ECFP fingerprints
y_tr   = np.load(y_train_in_fp)

X_val  = load_npz(X_val_ood_fp)      # val_ood ECFP fingerprints
y_val  = np.load(y_val_ood_fp)

X_test = load_npz(X_test_ood_fp)     # test_ood ECFP fingerprints
y_test = np.load(y_test_ood_fp)

print("Train_in shape:", X_tr.shape,  "| Positives:", int(y_tr.sum()))
print("Val_ood  shape:", X_val.shape, "| Positives:", int(y_val.sum()))
print("Test_ood shape:", X_test.shape,"| Positives:", int(y_test.sum()))

# -------------------------------------------
# STEP 1 — TRAIN RF ON train_in ONLY (EVAL ON val_ood)
# -------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

rf_clf.fit(X_tr, y_tr)
print("Random Forest trained on train_in.")

# ----- Validation on val_ood -----
val_proba_rf = rf_clf.predict_proba(X_val)[:, 1]

ap_val_rf  = average_precision_score(y_val, val_proba_rf)
roc_val_rf = roc_auc_score(y_val, val_proba_rf)

print(f"RF – Val_OOD AP:  {ap_val_rf:.6f}")
print(f"RF – Val_OOD AUC: {roc_val_rf:.6f}")

# ---------------------------------------------------------
# STEP 2 — RETRAIN ON TRAIN + VAL (train_in + val_ood)
# ---------------------------------------------------------
X_train_all = vstack([X_tr, X_val])
y_train_all = np.concatenate([y_tr, y_val])

print("Combined TRAIN (train_in + val_ood) shape:",
      X_train_all.shape, "| Positives:", int(y_train_all.sum()))

rf_final = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

rf_final.fit(X_train_all, y_train_all)
print("Random Forest retrained on TRAIN + VAL (train_in + val_ood).")

# -------------------------------------------
# STEP 3 — TEST ON test_ood
# -------------------------------------------
test_proba_rf = rf_final.predict_proba(X_test)[:, 1]

ap_test_rf  = average_precision_score(y_test, test_proba_rf)
roc_test_rf = roc_auc_score(y_test, test_proba_rf)

print(f"RF – Test_OOD AP:  {ap_test_rf:.6f}")
print(f"RF – Test_OOD AUC: {roc_test_rf:.6f}")
