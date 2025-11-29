"""
Process BELKA-style Parquet into sparse features for logistic regression (ridge/L2).

Repository layout note:
- This project uses a src/ layout with the package at: src/fit_proteins
- THIS script lives at: scripts/data_processed.py

Usage (from project root):
  # Direct file execution (no install needed):
  python scripts/data_processed.py --train
  python scripts/data_processed.py --test
  python scripts/data_processed.py --train --test

  # Override paths explicitly (optional):
  python scripts/data_processed.py --train_path data/raw/train_brd4_50k_stratified.parquet \
                                   --test_path  data/raw/test_brd4_50k.parquet
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
import joblib

# ---------- robust project root detection ----------
def _find_project_root(start: Path) -> Path:
    """
    Walk upwards from `start` to find the repo root
    (identified by `pyproject.toml` or `.git`). Falls back to parent.
    """
    cur = start
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # Fallback: assume <repo>/<scripts>/this_file.py
    return start.parent.parent

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
# --------------------------------------------------

# ---------- defaults aligned with your tree ----------
# data/
#   ├─ raw/
#   │   ├─ train_brd4_50k_stratified.parquet
#   │   └─ test_brd4_50k.parquet
#   └─ prep_logreg_ridge/  (outputs)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "raw" / "train_brd4_50k_stratified.parquet"
DEFAULT_TEST  = DEFAULT_DATA_DIR / "raw" / "test_brd4_50k.parquet"
OUT_DIR       = DEFAULT_DATA_DIR / "processed"
# ----------------------------------------------------



HASH_N_FEATURES = 2**18      # adjust to 2**19 or 2**20 if you have RAM/time
NGRAM_RANGE     = (2, 4)     # char n-grams for SMILES


def _mk_ohe():
    """Make OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _combine_smiles(df: pd.DataFrame) -> pd.Series:
    return (
        df["molecule_smiles"].fillna("") + " " +
        df["buildingblock1_smiles"].fillna("") + " " +
        df["buildingblock2_smiles"].fillna("") + " " +
        df["buildingblock3_smiles"].fillna("")
    )


def prepare_train(train_path: Path, out_dir: Path,
                  hash_n_features: int = HASH_N_FEATURES,
                  ngram_range = NGRAM_RANGE) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(train_path)

    need = {
        "id", "protein_name", "molecule_smiles",
        "buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles",
        "binds",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Training parquet missing columns: {miss}")

    df["binds"] = df["binds"].astype(int)
    df["smiles_all"] = _combine_smiles(df)

    vec = HashingVectorizer(
        n_features=hash_n_features,
        analyzer="char",
        ngram_range=ngram_range,
        alternate_sign=False,
        norm=None,
    )
    ohe = _mk_ohe().fit(df[["protein_name"]])

    Xs = vec.transform(df["smiles_all"].astype(str))
    Xp = ohe.transform(df[["protein_name"]])
    X  = hstack([Xs, Xp], format="csr")

    y   = df["binds"].to_numpy(dtype=np.int8)
    ids = df["id"].to_numpy()

    save_npz(out_dir / "X_train_full.npz", X)
    np.save(out_dir / "y_train_full.npy", y)
    np.save(out_dir / "ids_train_full.npy", ids)

    meta = {
        "hash_n_features": hash_n_features,
        "ngram_range": ngram_range,
        "train_path": str(train_path),
        "feature_blocks": ["hashed_smiles", "onehot_protein"],
    }
    joblib.dump({"ohe": ohe, "meta": meta}, out_dir / "prep_metadata.joblib")

    print(f"✅ Train prepared → {out_dir}")
    print("  X shape:", X.shape, "| positives:", int(y.sum()), f"({y.mean():.5f})")


def prepare_test(test_path: Path, out_dir: Path) -> None:
    art = joblib.load(out_dir / "prep_metadata.joblib")
    ohe  = art["ohe"]
    meta = art["meta"]

    vec = HashingVectorizer(
        n_features=int(meta["hash_n_features"]),
        analyzer="char",
        ngram_range=tuple(meta["ngram_range"]),
        alternate_sign=False,
        norm=None,
    )

    df = pd.read_parquet(test_path)
    need = {
        "id", "protein_name", "molecule_smiles",
        "buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Test parquet missing columns: {miss}")

    df["smiles_all"] = _combine_smiles(df)

    Xs = vec.transform(df["smiles_all"].astype(str))
    Xp = ohe.transform(df[["protein_name"]])  # unknown proteins safely ignored
    X  = hstack([Xs, Xp], format="csr")

    ids = df["id"].to_numpy()
    save_npz(out_dir / "X_test.npz", X)
    np.save(out_dir / "ids_test.npy", ids)

    print(f"✅ Test prepared → {out_dir}")
    print("  X_test shape:", X.shape)


def main():
    ap = argparse.ArgumentParser(description="Prepare train/test Parquet for logistic ridge.")
    ap.add_argument("--train", action="store_true", help="Prepare training parquet.")
    ap.add_argument("--test",  action="store_true", help="Prepare test parquet.")
    ap.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--test_path",  type=Path, default=DEFAULT_TEST)
    ap.add_argument("--out_dir",    type=Path, default=OUT_DIR)
    ap.add_argument("--hash_n_features", type=int, default=HASH_N_FEATURES)
    ap.add_argument("--ngram_min", type=int, default=NGRAM_RANGE[0])
    ap.add_argument("--ngram_max", type=int, default=NGRAM_RANGE[1])
    args = ap.parse_args()

    ngram_range = (args.ngram_min, args.ngram_max)

    if args.train:
        prepare_train(args.train_path, args.out_dir, args.hash_n_features, ngram_range)
    if args.test:
        prepare_test(args.test_path, args.out_dir)
    if not args.train and not args.test:
        ap.error("Pass --train and/or --test")


if __name__ == "__main__":
    main()