import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import sparse
from scipy.sparse import load_npz

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
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",      # fast for sparse, high-dimensional data
    n_estimators=225,       # fixed number of trees (no early stopping)
    learning_rate=0.05,
    max_depth=7,             # between 6–8 as per your instructions
    subsample=0.8,
    colsample_bytree=0.6,
    scale_pos_weight=scale_pos_weight,
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    eval_metric="aucpr",     # metric used during training (for logging only)
)

# 🔥 TRAIN ONLY ON TRAIN FILES
xgb_clf.fit(X_train_full, y_train_full)

print("XGBoost training complete.")
train_proba = xgb_clf.predict_proba(X_train_full)[:, 1]

ap_train = average_precision_score(y_train_full, train_proba)
roc_train = roc_auc_score(y_train_full, train_proba)

print(f"XGBoost – TRAIN AP:  {ap_train:.6f}")
print(f"XGBoost – TRAIN AUC: {roc_train:.6f}")
# 🔍 PREDICT ON TEST FILE (X_test)
test_proba = xgb_clf.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": ids_test,
    "binds": test_proba.astype(float),
})

# Create submission directory if it doesn't exist
submission_dir = Path("../../../data/submission_of_models")
submission_dir.mkdir(parents=True, exist_ok=True)

submission_path = submission_dir / "submission_xgb_fulltrain.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
submission.head()