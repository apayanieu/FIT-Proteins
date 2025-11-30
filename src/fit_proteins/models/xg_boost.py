#This script trains and evaluates an XGBoost classifier on BRD4 molecular fingerprint data.
#  The fingerprints are stored in sparse .npz format, which are efficiently loaded without 
# converting them to dense matrices. The goal is to measure how well XGBoost can generalize
#  to unseen data, so the training set is split into an 80% training subset and a 20% validation 
# subset, with stratification to preserve the positive/negative ratio in this highly imbalanced 
# dataset.

#A class-imbalance correction weight (scale_pos_weight = N_neg / N_pos) is computed to help XGBoost 
# deal with the low number of positive binders. The model is then trained using parameters suitable 
# for high-dimensional chemical fingerprint data, such as max_depth=7, subsampling, column sampling,
#  and the fast hist tree method. After training, predictions are produced for the 20% validation 
# split (data the model has never seen), and Average Precision (AP) and ROC-AUC are calculated to 
# assess real generalization performance. These metrics provide a realistic estimate of how XGBoost 
# will perform on new molecules and act as a benchmark before generating Kaggle submissions

# -------------------------------------------
# IMPORTS AND BASIC SETUP
# -------------------------------------------
import numpy as np
import pandas as pd
import os
from pathlib import Path

from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------------------------
# HELPER: CHECK FILE EXISTENCE
# -------------------------------------------
DATA_DIR = Path("../../../data/processed")

def ensure_exists(path):
    """
    Ensures that the required file exists.
    If missing, stops execution with a clear error message.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required file not found: {path!s}. "
            "Please place the file under data/processed or update the path."
        )

# -------------------------------------------
# FILE PATHS
# -------------------------------------------
X_train_full_fp = DATA_DIR / "X_train_full.npz"
X_test_fp       = DATA_DIR / "X_test.npz"
y_train_fp      = DATA_DIR / "y_train_full.npy"
ids_train_fp    = DATA_DIR / "ids_train_full.npy"
ids_test_fp     = DATA_DIR / "ids_test.npy"
splits_fp       = DATA_DIR / "train_brd4_50k_clean_blocks.parquet"

# Check all required files exist
for p in (X_train_full_fp, X_test_fp, y_train_fp, ids_train_fp, ids_test_fp, splits_fp):
    ensure_exists(p)

# -------------------------------------------
# LOAD DATA: FINGERPRINTS, LABELS, METADATA
# -------------------------------------------
# Load sparse fingerprint matrices (very efficient)
X_train_full = load_npz(X_train_full_fp)
X_test       = load_npz(X_test_fp)

# Load labels and molecule IDs
y_train_full   = np.load(y_train_fp)
ids_train_full = np.load(ids_train_fp)
ids_test       = np.load(ids_test_fp)

# Load BRD4 metadata (not used for this validation split)
splits_df = pd.read_parquet(splits_fp)

# Print shapes for sanity check
print("X_train_full shape:", X_train_full.shape)
print("X_test shape:", X_test.shape)
print("y_train_full shape:", y_train_full.shape)
print("ids_train_full shape:", ids_train_full.shape)
print("ids_test shape:", ids_test.shape)
print("Splits columns:", splits_df.columns.tolist())

# -------------------------------------------
# CALCULATE CLASS IMBALANCE WEIGHT
# -------------------------------------------
# Strong imbalance: very few positive binders
N_pos = int((y_train_full == 1).sum())
N_neg = int((y_train_full == 0).sum())
scale_pos_weight = N_neg / N_pos

print(f"N_pos (train): {N_pos}")
print(f"N_neg (train): {N_neg}")
print(f"scale_pos_weight = N_neg / N_pos = {scale_pos_weight:.2f}")

# -------------------------------------------
# TRAIN/VALIDATION SPLIT (80% / 20%)
# -------------------------------------------
# Split into training and validation; stratification keeps class ratios
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("Train shape:", X_tr.shape, " | Positives:", y_tr.sum())
print("Val   shape:", X_val.shape, " | Positives:", y_val.sum())

# -------------------------------------------
# DEFINE AND TRAIN XGBOOST MODEL
# -------------------------------------------
# XGBoost settings chosen for sparse ECFP fingerprints
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",      # Faster on large sparse matrices
    n_estimators=500,        # Number of trees
    learning_rate=0.05,
    max_depth=7,             # Recommended depth (6–8)
    subsample=0.8,           # Row sampling for regularization
    colsample_bytree=0.6,    # Column sampling for regularization
    scale_pos_weight=scale_pos_weight,  # Correct class imbalance
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_jobs=-1,               # Use all CPU cores
    random_state=RANDOM_STATE,
    eval_metric="aucpr"      # Optimizes ranking (like AP)
)

# Train the model on the 80% training split
xgb_clf.fit(X_tr, y_tr)
print("XGBoost trained.")

# -------------------------------------------
# VALIDATION EVALUATION (UNSEEN DATA)
# -------------------------------------------
# Predict probabilities on the 20% validation set
val_proba_xgb = xgb_clf.predict_proba(X_val)[:, 1]

# Compute AP and AUC on data the model never saw
ap_val_xgb = average_precision_score(y_val, val_proba_xgb)
roc_val_xgb = roc_auc_score(y_val, val_proba_xgb)

print(f"XGBoost – Validation AP:  {ap_val_xgb:.6f}")
print(f"XGBoost – Validation AUC: {roc_val_xgb:.6f}")