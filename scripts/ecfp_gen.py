from __future__ import annotations

"""
Step 3: ECFP feature generation for internal OOD splits (train_in / val_ood / test_ood).

Inputs (default paths match chemistry_pipeline.py outputs):
  data/processed/train_brd4_50k_clean_blocks_train.parquet
  data/processed/train_brd4_50k_clean_blocks_val.parquet
  data/processed/train_brd4_50k_clean_blocks_test.parquet

Features:
  - Morgan/ECFP on 'smiles_clean'
  - radius=2, n_bits=2048, use_chirality=True (configurable)

Outputs (easy, unambiguous names):
  X_train_in_ecfp.npz   y_train_in.npy   ids_train_in.npy
  X_val_ood_ecfp.npz    y_val_ood.npy    ids_val_ood.npy
  X_test_ood_ecfp.npz   y_test_ood.npy   ids_test_ood.npy
  ecfp_metadata.joblib

This script overwrites outputs on rerun.
"""

import argparse
from pathlib import Path
from typing import Iterable, Dict, Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import sparse

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

def get_project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()

PROJECT_ROOT  = get_project_root()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IN_DEFAULT = PROCESSED_DIR / "train_brd4_50k_clean_blocks_train.parquet"
VAL_IN_DEFAULT   = PROCESSED_DIR / "train_brd4_50k_clean_blocks_val.parquet"
TEST_IN_DEFAULT  = PROCESSED_DIR / "train_brd4_50k_clean_blocks_test.parquet"

# Outputs (distinct, readable names)
X_TRAIN_OUT = PROCESSED_DIR / "X_train_in_ecfp.npz"
Y_TRAIN_OUT = PROCESSED_DIR / "y_train_in.npy"
IDS_TRAIN_OUT = PROCESSED_DIR / "ids_train_in.npy"

X_VAL_OUT = PROCESSED_DIR / "X_val_ood_ecfp.npz"
Y_VAL_OUT = PROCESSED_DIR / "y_val_ood.npy"
IDS_VAL_OUT = PROCESSED_DIR / "ids_val_ood.npy"

X_TEST_OUT = PROCESSED_DIR / "X_test_ood_ecfp.npz"
Y_TEST_OUT = PROCESSED_DIR / "y_test_ood.npy"
IDS_TEST_OUT = PROCESSED_DIR / "ids_test_ood.npy"

META_OUT   = PROCESSED_DIR / "ecfp_metadata.joblib"

REQUIRED_COLS = {"id", "smiles_clean", "binds"}

# --------------------------------------------------------------------------------------
# ECFP helpers
# --------------------------------------------------------------------------------------

def smiles_to_ecfp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> sparse.csr_matrix:
    """Return a 1 x n_bits CSR row; all-zero if SMILES is invalid."""
    if not isinstance(smiles, str):
        return sparse.csr_matrix((1, n_bits), dtype=np.int8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return sparse.csr_matrix((1, n_bits), dtype=np.int8)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, useChirality=use_chirality
    )
    arr = np.zeros((1, n_bits), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr[0])
    return sparse.csr_matrix(arr)

def compute_ecfp_matrix(
    smiles_list: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
    verbose: bool = True,
) -> sparse.csr_matrix:
    rows = []
    for i, s in enumerate(smiles_list):
        if verbose and (i + 1) % 5000 == 0:
            print(f"  processed {i + 1} molecules...")
        rows.append(smiles_to_ecfp(s, radius=radius, n_bits=n_bits, use_chirality=use_chirality))
    X = sparse.vstack(rows, format="csr")
    X.data = X.data.astype(np.int8, copy=False)
    return X

# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------

def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    df = pd.read_parquet(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} missing columns: {missing}")
    # enforce dtypes
    df = df.copy()
    df["binds"] = df["binds"].astype(int)
    return df

def save_split(
    X: sparse.csr_matrix, y: np.ndarray, ids: np.ndarray,
    X_path: Path, y_path: Path, ids_path: Path
) -> None:
    sparse.save_npz(X_path, X)
    np.save(y_path, y.astype(np.int8, copy=False))
    np.save(ids_path, ids)

# --------------------------------------------------------------------------------------
# Sanity reporting
# --------------------------------------------------------------------------------------

def report_split(name: str, X: sparse.csr_matrix, y: np.ndarray) -> None:
    pos_rate = float(y.mean()) if y.size else float("nan")
    print(f"[{name}] X shape: {X.shape} | positives: {int(y.sum())}/{len(y)} ({pos_rate:.5f}) "
          f"| avg bits on: {X.nnz / max(1, X.shape[0]):.1f}")

def rf_sanity(train_X, train_y, val_X, val_y) -> None:
    """Tiny RF check: train on train_in, score AP on val_ood."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score

        print("\n[rf] Quick sanity: RF on train_in → AP on val_ood")
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
        )
        rf.fit(train_X, train_y)
        proba = rf.predict_proba(val_X)[:, 1]
        ap = average_precision_score(val_y, proba)
        print(f"[rf] AP(val_ood) = {ap:.4f} (baseline sanity only)")
    except Exception as e:
        print("[rf] Skipping RF sanity check:", repr(e))

# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    radius = args.radius
    n_bits = args.n_bits
    use_chirality = not args.no_chirality

    train_p = Path(args.train_in or TRAIN_IN_DEFAULT)
    val_p   = Path(args.val_in or VAL_IN_DEFAULT)
    test_p  = Path(args.test_in or TEST_IN_DEFAULT)
    out_dir = Path(args.out_dir or PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[read]", train_p.name, "|", val_p.name, "|", test_p.name)
    df_tr = load_split(train_p)
    df_va = load_split(val_p)
    df_te = load_split(test_p)

    print("\n[ecfp] Parameters:",
          f"radius={radius}  n_bits={n_bits}  use_chirality={use_chirality}")

    # Compute fingerprints per split
    print("\n[ecfp] Computing train_in...")
    X_tr = compute_ecfp_matrix(df_tr["smiles_clean"].astype(str), radius, n_bits, use_chirality, verbose=True)
    y_tr = df_tr["binds"].to_numpy()
    ids_tr = df_tr["id"].to_numpy()

    print("[ecfp] Computing val_ood...")
    X_va = compute_ecfp_matrix(df_va["smiles_clean"].astype(str), radius, n_bits, use_chirality, verbose=True)
    y_va = df_va["binds"].to_numpy()
    ids_va = df_va["id"].to_numpy()

    print("[ecfp] Computing test_ood...")
    X_te = compute_ecfp_matrix(df_te["smiles_clean"].astype(str), radius, n_bits, use_chirality, verbose=True)
    y_te = df_te["binds"].to_numpy()
    ids_te = df_te["id"].to_numpy()

    # Save artifacts with clear names
    print("\n[write] Saving artifacts to:", out_dir)
    save_split(X_tr, y_tr, ids_tr, out_dir / X_TRAIN_OUT.name, out_dir / Y_TRAIN_OUT.name, out_dir / IDS_TRAIN_OUT.name)
    save_split(X_va, y_va, ids_va, out_dir / X_VAL_OUT.name,   out_dir / Y_VAL_OUT.name,   out_dir / IDS_VAL_OUT.name)
    save_split(X_te, y_te, ids_te, out_dir / X_TEST_OUT.name,  out_dir / Y_TEST_OUT.name,  out_dir / IDS_TEST_OUT.name)

    # Metadata for reproducibility
    meta: Dict[str, Any] = {
        "ecfp": {"radius": int(radius), "n_bits": int(n_bits), "use_chirality": bool(use_chirality)},
        "sources": {
            "train_in": str(train_p.resolve()),
            "val_ood":  str(val_p.resolve()),
            "test_ood": str(test_p.resolve()),
        },
        "outputs": {
            "X_train_in": str((out_dir / X_TRAIN_OUT.name).resolve()),
            "y_train_in": str((out_dir / Y_TRAIN_OUT.name).resolve()),
            "ids_train_in": str((out_dir / IDS_TRAIN_OUT.name).resolve()),
            "X_val_ood": str((out_dir / X_VAL_OUT.name).resolve()),
            "y_val_ood": str((out_dir / Y_VAL_OUT.name).resolve()),
            "ids_val_ood": str((out_dir / IDS_VAL_OUT.name).resolve()),
            "X_test_ood": str((out_dir / X_TEST_OUT.name).resolve()),
            "y_test_ood": str((out_dir / Y_TEST_OUT.name).resolve()),
            "ids_test_ood": str((out_dir / IDS_TEST_OUT.name).resolve()),
        }
    }
    joblib.dump(meta, out_dir / META_OUT.name)
    print("[write]", META_OUT.name)

    # Sanity summaries
    print("\n================ Sanity checks ================")
    report_split("train_in", X_tr, y_tr)
    report_split("val_ood",  X_va, y_va)
    report_split("test_ood", X_te, y_te)
    rf_sanity(X_tr, y_tr, X_va, y_va)
    print("============== End sanity checks ==============\n")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ECFP features for block-OOD splits (train_in/val_ood/test_ood).")
    p.add_argument("--train-in", type=str, default=None,
                   help="Path to *_clean_blocks_train.parquet")
    p.add_argument("--val-in",   type=str, default=None,
                   help="Path to *_clean_blocks_val.parquet")
    p.add_argument("--test-in",  type=str, default=None,
                   help="Path to *_clean_blocks_test.parquet")
    p.add_argument("--out-dir",  type=str, default=None,
                   help="Output directory (default: data/processed)")
    p.add_argument("--radius",   type=int, default=2, help="ECFP radius")
    p.add_argument("--n-bits",   dest="n_bits", type=int, default=2048, help="ECFP bit length")
    p.add_argument("--no-chirality", action="store_true", help="Disable chirality (default: enabled)")
    return p

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()
