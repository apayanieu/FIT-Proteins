#This script trains and evaluates a Random Forest classifier on molecular fingerprint data 
# from the BRD4 dataset. The data consist of sparse ECFP-like fingerprint vectors stored in .npz
#  format and corresponding binary activity labels. To properly measure how well the model general
# izes, the training dataset is split into an 80% training subset and a 20% validation subset using 
# stratification to preserve the imbalance ratio.

#The Random Forest model is configured with class balancing (class_weight="balanced") and max_features="sqrt", which is standard
#  for high-dimensional fingerprint data. After training on the 80% subset, the model predicts activity scores on the untouched 20% validation subset. 
# We compute Average Precision (AP) and ROC-AUC to evaluate how well the model performs on unseen data. These metrics reflect the model’s ranking quality 
# and discrimination power and give insight into real-world performance before submitting predictions to Kaggle.
# -------------------------------------------
# IMPORTS AND SETUP
# -------------------------------------------
import numpy as np
import pandas as pd
import os

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------------------------
# ENSURE DATA FILES EXIST
# -------------------------------------------
DATA_DIR = Path("../../../data/processed")

def ensure_exists(path):
    """
    Helper function to verify that a required file exists.
    Raises an error with a clear message if not found.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path!s}. "
                                "Add the file to the repo under data/processed or adjust the path.")

# File paths
X_train_full_fp = DATA_DIR / "X_train_full.npz"
X_test_fp       = DATA_DIR / "X_test.npz"
y_train_fp      = DATA_DIR / "y_train_full.npy"
ids_train_fp    = DATA_DIR / "ids_train_full.npy"
ids_test_fp     = DATA_DIR / "ids_test.npy"
splits_fp       = DATA_DIR / "train_brd4_50k_clean_blocks.parquet"

# Check all required files
for p in (X_train_full_fp, X_test_fp, y_train_fp, ids_train_fp, ids_test_fp, splits_fp):
    ensure_exists(p)

# -------------------------------------------
# LOAD FINGERPRINTS, LABELS & META
# -------------------------------------------
# Load sparse feature matrices (ECFP fingerprints)
X_train_full = load_npz(X_train_full_fp)
X_test       = load_npz(X_test_fp)

# Load activity labels and molecule IDs
y_train_full   = np.load(y_train_fp)
ids_train_full = np.load(ids_train_fp)
ids_test       = np.load(ids_test_fp)

# Load BRD4 split metadata (not used for this split but validated for completeness)
splits_df = pd.read_parquet(splits_fp)

print("X_train_full shape:", X_train_full.shape)
print("X_test shape:", X_test.shape)
print("y_train_full shape:", y_train_full.shape)
print("ids_train_full shape:", ids_train_full.shape)
print("ids_test shape:", ids_test.shape)
print("Splits columns:", splits_df.columns.tolist())

# -------------------------------------------
# TRAIN / VALIDATION SPLIT (80/20)
# -------------------------------------------
# Split the data into training (80%) and validation (20%)
# Using stratify=y_train_full preserves the % of active/inactive molecules
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
# TRAIN RANDOM FOREST CLASSIFIER
# -------------------------------------------
# Random Forest with settings suitable for large sparse chemical fingerprints
rf_clf = RandomForestClassifier(
    n_estimators=250,
    max_features="sqrt",    # Standard for high-dimensional sparse inputs
    max_depth=None,         # Fully grown trees (can be changed if overfitting)
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced", # Corrects for heavy class imbalance
    random_state=RANDOM_STATE,
)

rf_clf.fit(X_tr, y_tr)
print("Random Forest trained.")

# -------------------------------------------
# VALIDATION EVALUATION (MODEL HAS NEVER SEEN THIS DATA)
# -------------------------------------------
# Predict probabilities on validation set
val_proba_rf = rf_clf.predict_proba(X_val)[:, 1]

# Compute AP and AUC on unseen data
ap_val_rf = average_precision_score(y_val, val_proba_rf)
roc_val_rf = roc_auc_score(y_val, val_proba_rf)

print(f"RF – Validation AP:  {ap_val_rf:.6f}")
print(f"RF – Validation AUC: {roc_val_rf:.6f}")