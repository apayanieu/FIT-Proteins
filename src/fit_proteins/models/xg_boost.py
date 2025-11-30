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
# XGBOOST WITH OFFICIAL BLOCK-AWARE TRAIN/VAL SPLIT + FULL RETRAIN
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from scipy.sparse import load_npz
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

X_train_full_fp = DATA_DIR / "X_train_full.npz"
X_test_fp       = DATA_DIR / "X_test.npz"
y_train_fp      = DATA_DIR / "y_train_full.npy"
ids_train_fp    = DATA_DIR / "ids_train_full.npy"
ids_test_fp     = DATA_DIR / "ids_test.npy"
splits_fp       = DATA_DIR / "train_brd4_50k_clean_blocks.parquet"

for p in (X_train_full_fp, X_test_fp, y_train_fp, ids_train_fp, ids_test_fp, splits_fp):
    ensure_exists(p)

# -------------------------------------------
# LOAD FEATURES, LABELS, IDS, AND SPLIT METADATA
# -------------------------------------------
X_train_full = load_npz(X_train_full_fp)
X_test       = load_npz(X_test_fp)

y_train_full   = np.load(y_train_fp)
ids_train_full = np.load(ids_train_fp)
ids_test       = np.load(ids_test_fp)

splits_df = pd.read_parquet(splits_fp)

print("X_train_full:", X_train_full.shape)
print("X_test:", X_test.shape)
print("y_train_full:", y_train_full.shape)
print("splits columns:", splits_df.columns.tolist())

# -------------------------------------------
# APPLY OFFICIAL BLOCK-AWARE SPLIT
# -------------------------------------------
df = pd.DataFrame({"id": ids_train_full})
df = df.merge(splits_df[["id", "split_group"]], on="id", how="left")

train_mask = df["split_group"] == "train_in"
val_mask   = df["split_group"] == "val_ood"

X_tr = X_train_full[train_mask.values]
y_tr = y_train_full[train_mask.values]

X_val = X_train_full[val_mask.values]
y_val = y_train_full[val_mask.values]

print("train_in:", X_tr.shape, "| positives:", y_tr.sum())
print("val_ood:", X_val.shape, "| positives:", y_val.sum())

# -------------------------------------------
# CLASS IMBALANCE WEIGHT (REQUIRED FOR XGBOOST)
# -------------------------------------------
N_pos = int((y_tr == 1).sum())
N_neg = int((y_tr == 0).sum())
scale_pos_weight = N_neg / N_pos

print("scale_pos_weight (train_in):", scale_pos_weight)

# -------------------------------------------
# STEP 1 — TRAIN XGBOOST ON train_in ONLY
# -------------------------------------------
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    n_estimators=500,
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

# -------------------------------------------
# STEP 2 — VALIDATE ON val_ood (UNSEEN BUILDING BLOCKS)
# -------------------------------------------
val_proba_xgb = xgb_clf.predict_proba(X_val)[:, 1]

ap_val_xgb  = average_precision_score(y_val, val_proba_xgb)
roc_val_xgb = roc_auc_score(y_val, val_proba_xgb)

print(f"XGBoost – Val_OOD AP:  {ap_val_xgb:.6f}")
print(f"XGBoost – Val_OOD AUC: {roc_val_xgb:.6f}")

# ---------------------------------------------------------
# STEP 3 — RETRAIN XGBOOST ON ALL DATA (train_in + val_ood)
# ---------------------------------------------------------
# This final model will then be used for predicting X_test.
xgb_final = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.6,
    scale_pos_weight=scale_pos_weight,   # still computed from train_in
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    eval_metric="aucpr"
)

xgb_final.fit(X_train_full, y_train_full)
print("XGBoost retrained on FULL training set.")
