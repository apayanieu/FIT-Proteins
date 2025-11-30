import numpy as np
import pandas as pd
import os

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path("../../../data/processed")
def ensure_exists(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path!s}. "
                                "Add the file to the repo under data/processed or set FIT_DATA_DIR to its location.")

# Features (use Path objects)
X_train_full_fp = DATA_DIR / "X_train_full.npz"
X_test_fp       = DATA_DIR / "X_test.npz"
y_train_fp      = DATA_DIR / "y_train_full.npy"
ids_train_fp    = DATA_DIR / "ids_train_full.npy"
ids_test_fp     = DATA_DIR / "ids_test.npy"
splits_fp       = DATA_DIR / "train_brd4_50k_clean_blocks.parquet"

for p in (X_train_full_fp, X_test_fp, y_train_fp, ids_train_fp, ids_test_fp, splits_fp):
    ensure_exists(p)

# Load sparse matrices properly
X_train_full = load_npz(X_train_full_fp)
X_test       = load_npz(X_test_fp)

# Labels & ids
y_train_full   = np.load(y_train_fp)
ids_train_full = np.load(ids_train_fp)
ids_test       = np.load(ids_test_fp)

# Splits for BRD4 (train_in / val_ood)
splits_df = pd.read_parquet(splits_fp)

print("X_train_full shape:", X_train_full.shape)
print("X_test shape:", X_test.shape)
print("y_train_full shape:", y_train_full.shape)
print("ids_train_full shape:", ids_train_full.shape)
print("ids_test shape:", ids_test.shape)
print("Splits columns:", splits_df.columns.tolist())

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,         # 20% validation
    random_state=RANDOM_STATE,
    stratify=y_train_full   # keeps the same class ratio
)

print("Train shape:", X_tr.shape, " | Positives:", y_tr.sum())
print("Val   shape:", X_val.shape, " | Positives:", y_val.sum())

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
print("Random Forest trained.")
#Validation
val_proba_rf = rf_clf.predict_proba(X_val)[:, 1]

ap_val_rf = average_precision_score(y_val, val_proba_rf)
roc_val_rf = roc_auc_score(y_val, val_proba_rf)

print(f"RF – Validation AP:  {ap_val_rf:.6f}")
print(f"RF – Validation AUC: {roc_val_rf:.6f}")