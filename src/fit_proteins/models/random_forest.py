import numpy as np
import pandas as pd
import os

from pathlib import Path
from scipy.sparse import load_npz

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

rf_clf = RandomForestClassifier(
    n_estimators=200,     # number of trees
    max_features="sqrt",  # √d (correct parameter, not max_depth)
    max_depth=None,       # fully grown trees; you can set e.g. 20 if too slow
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
)

# 🔥 TRAIN ONLY ON TRAINING FILES
rf_clf.fit(X_train_full, y_train_full)

print("Random Forest training complete.")

train_proba = rf_clf.predict_proba(X_train_full)[:, 1]

ap_train = average_precision_score(y_train_full, train_proba)
roc_train = roc_auc_score(y_train_full, train_proba)

print(f"Random Forest – TRAIN AP:  {ap_train:.6f}")
print(f"Random Forest – TRAIN AUC: {roc_train:.6f}")
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