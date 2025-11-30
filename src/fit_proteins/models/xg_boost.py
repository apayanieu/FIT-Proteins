import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

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

N_pos = int((y_train_full == 1).sum())
N_neg = int((y_train_full == 0).sum())
scale_pos_weight = N_neg / N_pos

print(f"N_pos (train): {N_pos}")
print(f"N_neg (train): {N_neg}")
print(f"scale_pos_weight = N_neg / N_pos = {scale_pos_weight:.2f}")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,         # 20% validation
    random_state=RANDOM_STATE,
    stratify=y_train_full   # keeps the same class ratio
)
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,          # 6 to 8 recommended
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
print("XGBoost trained.")
val_proba_xgb = xgb_clf.predict_proba(X_val)[:, 1]

ap_val_xgb = average_precision_score(y_val, val_proba_xgb)
roc_val_xgb = roc_auc_score(y_val, val_proba_xgb)

print(f"XGBoost – Validation AP:  {ap_val_xgb:.6f}")
print(f"XGBoost – Validation AUC: {roc_val_xgb:.6f}")
