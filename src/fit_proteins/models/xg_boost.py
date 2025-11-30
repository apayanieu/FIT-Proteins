"""
This script trains, evaluates, and finalizes an XGBoost classifier for BRD4 binding prediction
using the official block-aware train/validation split provided in the project metadata
(`train_brd4_50k_clean_blocks.parquet`).

The dataset consists of high-dimensional ECFP-style molecular fingerprints stored as sparse
matrices. The block-aware split divides molecules into:
    - train_in: molecules built from building blocks seen during training
    - val_ood: molecules containing at least one building block never seen in training

This out-of-distribution (OOD) split is essential for evaluating true generalization to novel
chemical fragments, as required by the project design.

The script performs the following steps:
1. Loads ECFP fingerprints, labels, molecule IDs, and the official train_in/val_ood metadata.
2. Separates the training data into train_in (for fitting) and val_ood (for OOD evaluation).
3. Computes the scale_pos_weight parameter to correct for extreme class imbalance.
4. Trains an XGBoost model *only on train_in* and evaluates it on val_ood using:
       - Average Precision (AP): the primary metric
       - ROC-AUC: overall separability
5. After evaluation, the model is retrained on all available training data
   (train_in + val_ood) to maximize learning before generating test-set predictions.

This workflow follows the required BRD4 pipeline:
    - OOD evaluation using val_ood (Step 2 in the project instructions)
    - Retraining on full data before test inference (Step 10 in the project instructions)
and ensures no leakage between training and OOD validation.
"""

# ============================================================
# XGBOOST USING PRE-SPLIT ECFP DATA (TRAIN_IN / VAL_OOD / TEST_OOD)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from scipy.sparse import load_npz, vstack
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------------------------
# ENSURE FILES EXIST
# -------------------------------------------
DATA_DIR = Path("../../../data/processed")

def ensure_exists(path):
    """Check file existence and raise a clear error if missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing required file: {path!s}. "
            "Place it under data/processed or update the path."
        )

# Pre-split ECFP files (all in the same folder)
X_train_in_fp = DATA_DIR / "X_train_in_ecfp.npz"
y_train_in_fp = DATA_DIR / "y_train_in.npy"

X_val_ood_fp  = DATA_DIR / "X_val_ood_ecfp.npz"
y_val_ood_fp  = DATA_DIR / "y_val_ood.npy"

X_test_ood_fp = DATA_DIR / "X_test_ood_ecfp.npz"
y_test_ood_fp = DATA_DIR / "y_test_ood.npy"

for p in (X_train_in_fp, y_train_in_fp,
          X_val_ood_fp,  y_val_ood_fp,
          X_test_ood_fp, y_test_ood_fp):
    ensure_exists(p)

# -------------------------------------------
# LOAD PRE-SPLIT DATA
# -------------------------------------------
X_tr   = load_npz(X_train_in_fp)   # train_in fingerprints
y_tr   = np.load(y_train_in_fp)

X_val  = load_npz(X_val_ood_fp)    # val_ood fingerprints
y_val  = np.load(y_val_ood_fp)

X_test = load_npz(X_test_ood_fp)   # test_ood fingerprints
y_test = np.load(y_test_ood_fp)

print("Train_in shape:", X_tr.shape,  "| Positives:", int(y_tr.sum()))
print("Val_ood  shape:", X_val.shape, "| Positives:", int(y_val.sum()))
print("Test_ood shape:", X_test.shape,"| Positives:", int(y_test.sum()))

# -------------------------------------------
# CLASS IMBALANCE WEIGHT (FROM train_in)
# -------------------------------------------
N_pos = int((y_tr == 1).sum())
N_neg = int((y_tr == 0).sum())
scale_pos_weight = N_neg / N_pos

print("scale_pos_weight (train_in):", scale_pos_weight)

# -------------------------------------------
# STEP 1 — TRAIN XGBOOST ON train_in ONLY (for val_ood evaluation)
# -------------------------------------------
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.6,
    scale_pos_weight=scale_pos_weight,
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    eval_metric="aucpr"
)

xgb_clf.fit(X_tr, y_tr)
print("XGBoost trained on train_in.")

# ----- Validation on val_ood -----
val_proba_xgb = xgb_clf.predict_proba(X_val)[:, 1]

ap_val_xgb  = average_precision_score(y_val, val_proba_xgb)
roc_val_xgb = roc_auc_score(y_val, val_proba_xgb)

print(f"XGBoost – Val_OOD AP:  {ap_val_xgb:.6f}")
print(f"XGBoost – Val_OOD AUC: {roc_val_xgb:.6f}")

# ---------------------------------------------------------
# STEP 2 — RETRAIN ON TRAIN + VAL (train_in + val_ood)
# ---------------------------------------------------------
X_train_all = vstack([X_tr, X_val])
y_train_all = np.concatenate([y_tr, y_val])

print("Combined TRAIN (train_in + val_ood) shape:",
      X_train_all.shape, "| Positives:", int(y_train_all.sum()))

xgb_final = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.6,
    scale_pos_weight=scale_pos_weight,  # still based on train_in
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    eval_metric="aucpr"
)

xgb_final.fit(X_train_all, y_train_all)
print("XGBoost retrained on TRAIN + VAL (train_in + val_ood).")

# -------------------------------------------
# STEP 3 — TEST ON test_ood
# -------------------------------------------
test_proba_xgb = xgb_final.predict_proba(X_test)[:, 1]

ap_test_xgb  = average_precision_score(y_test, test_proba_xgb)
roc_test_xgb = roc_auc_score(y_test, test_proba_xgb)

print(f"XGBoost – Test_OOD AP:  {ap_test_xgb:.6f}")
print(f"XGBoost – Test_OOD AUC: {roc_test_xgb:.6f}")
