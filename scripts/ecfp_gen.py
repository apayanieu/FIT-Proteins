from __future__ import annotations

"""
Step 3: Feature generation (ECFP baseline) for BELKA–BRD4.

What this script does
---------------------
1) Read block-processed chemistry files:
     data/processed/train_brd4_50k_clean_blocks.parquet
     data/processed/test_brd4_50k_clean_blocks.parquet

2) Compute Morgan/ECFP fingerprints on `smiles_clean` with:
     - radius = 2
     - n_bits = 2048
     - use_chirality = True

3) Persist artifacts for modeling:
     - X_train_full.npz  (CSR sparse matrix, 50k x 2048)
     - X_test.npz        (CSR sparse matrix, 50k x 2048)
     - y_train_full.npy  (binds labels)
     - ids_train_full.npy
     - ids_test.npy
     - prep_metadata.joblib (fingerprint parameters + sources)

4) Run sanity checks:
     - Check shapes and alignment of all arrays
     - Check class balance (positive rate) in y_train_full
     - Check fingerprint sparsity (avg bits=1 per molecule, random rows)
     - OPTIONAL: quick RandomForest sanity AP (if scikit-learn is installed)

You can re-run this script safely; it overwrites outputs.
"""

import argparse
from pathlib import Path
from typing import Optional, Iterable, Dict, Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import sparse


# --------------------------------------------------------------------------------------
# Paths & utilities
# --------------------------------------------------------------------------------------


def get_project_root() -> Path:
    """Compute project root whether run as script or interactively."""
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        # __file__ not defined (e.g., interactive / notebook). Fall back to CWD.
        return Path.cwd()


PROJECT_ROOT = get_project_root()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Default IO paths (matched to chemistry_pipeline outputs)
TRAIN_BLOCKS_IN = PROCESSED_DIR / "train_brd4_50k_clean_blocks.parquet"
TEST_BLOCKS_IN = PROCESSED_DIR / "test_brd4_50k_clean_blocks.parquet"

X_TRAIN_OUT = PROCESSED_DIR / "X_train_full.npz"
X_TEST_OUT = PROCESSED_DIR / "X_test.npz"
Y_TRAIN_OUT = PROCESSED_DIR / "y_train_full.npy"
IDS_TRAIN_OUT = PROCESSED_DIR / "ids_train_full.npy"
IDS_TEST_OUT = PROCESSED_DIR / "ids_test.npy"
META_OUT = PROCESSED_DIR / "prep_metadata.joblib"


REQUIRED_TRAIN_COLS = {"id", "smiles_clean", "binds"}
REQUIRED_TEST_COLS = {"id", "smiles_clean"}


# --------------------------------------------------------------------------------------
# Core ECFP helpers
# --------------------------------------------------------------------------------------


def smiles_to_ecfp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> sparse.csr_matrix:
    """
    Convert a SMILES string into a 1 x n_bits ECFP fingerprint (CSR row).

    - If RDKit fails to parse, returns an all-zero row.
    """
    if not isinstance(smiles, str):
        return sparse.csr_matrix((1, n_bits), dtype=np.int8)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Invalid SMILES -> all zeros; keeps row count aligned
        return sparse.csr_matrix((1, n_bits), dtype=np.int8)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
        useChirality=use_chirality,
    )

    arr = np.zeros((1, n_bits), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr[0])
    return sparse.csr_matrix(arr)


def compute_ecfp_matrix(
    smiles_list: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
    verbose: bool = True,
) -> sparse.csr_matrix:
    """
    Compute a CSR matrix of ECFP fingerprints for a list of SMILES.

    Returns: (n_mols, n_bits) CSR matrix with dtype int8.
    """
    rows = []
    for i, s in enumerate(smiles_list):
        if verbose and (i + 1) % 5000 == 0:
            print(f"  processed {i + 1} molecules...")
        rows.append(smiles_to_ecfp(s, radius=radius, n_bits=n_bits, use_chirality=use_chirality))

    X = sparse.vstack(rows, format="csr")
    X.data = X.data.astype(np.int8, copy=False)
    return X


# --------------------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------------------


def run_sanity_checks(
    X_train: sparse.csr_matrix,
    X_test: sparse.csr_matrix,
    y_train: np.ndarray,
    ids_train: np.ndarray,
    ids_test: np.ndarray,
) -> None:
    """Print basic sanity checks to stdout."""
    print("\n================ Sanity checks ================")

    # Shape checks
    print("[shape] X_train:", X_train.shape)
    print("[shape] X_test :", X_test.shape)
    print("[shape] y_train:", y_train.shape)
    print("[shape] ids_train:", ids_train.shape)
    print("[shape] ids_test :", ids_test.shape)

    assert X_train.shape[0] == y_train.shape[0] == ids_train.shape[0], \
        "Row count mismatch between X_train, y_train, ids_train"
    assert X_test.shape[0] == ids_test.shape[0], \
        "Row count mismatch between X_test, ids_test"

    # Label distribution
    pos_rate = float(y_train.mean())
    n_pos = int(y_train.sum())
    n_tot = int(y_train.shape[0])
    print("\n[label] Total train molecules:", n_tot)
    print("[label] Positives:", n_pos)
    print("[label] Positive rate:", f"{pos_rate:.4f}")

    # Fingerprint sparsity
    avg_bits_on_train = X_train.nnz / X_train.shape[0]
    avg_bits_on_test = X_test.nnz / X_test.shape[0]
    print("\n[fp] Average #bits=1 per molecule (train):", f"{avg_bits_on_train:.1f}")
    print("[fp] Average #bits=1 per molecule (test) :", f"{avg_bits_on_test:.1f}")

    rng = np.random.default_rng(42)
    rows = rng.choice(X_train.shape[0], size=min(5, X_train.shape[0]), replace=False)
    print("\n[fp] Example rows (train):")
    for r in rows:
        n_on = X_train[r].nnz
        print(f"  row {int(r)}: bits on = {n_on}")

    # Optional: quick RF sanity AP if sklearn installed
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import train_test_split

        print("\n[rf] Running quick RandomForest sanity check (this is NOT the final model)...")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )

        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_tr, y_tr)
        val_scores = rf.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, val_scores)
        print(f"[rf] Sanity-check AP on random 20% split: {ap:.4f}")
        print("[rf] (If this is > positive rate, features carry some signal.)")
    except ImportError:
        print("\n[rf] sklearn not installed, skipping RF sanity check.")
    except Exception as e:
        print("\n[rf] RF sanity check failed with error:")
        print("    ", repr(e))

    print("============== End sanity checks ==============\n")


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------


def run_ecfp(args: argparse.Namespace) -> None:
    """
    Orchestrate Step 3: read Parquets, compute ECFP features, save artifacts, run sanity checks.
    """
    radius = args.radius
    n_bits = args.n_bits
    use_chirality = not args.no_chirality

    # Resolve input paths (allow overrides)
    train_path = Path(args.train_in or TRAIN_BLOCKS_IN)
    test_path = Path(args.test_in or TEST_BLOCKS_IN)

    print("[read] train:", train_path)
    print("[read] test :", test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test parquet not found: {test_path}")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Basic column checks
    missing_train = REQUIRED_TRAIN_COLS - set(train_df.columns)
    missing_test = REQUIRED_TEST_COLS - set(test_df.columns)
    if missing_train:
        raise KeyError(f"Train parquet missing required columns: {missing_train}")
    if missing_test:
        raise KeyError(f"Test parquet missing required columns: {missing_test}")

    # Extract SMILES, labels, IDs
    train_smiles = train_df["smiles_clean"].astype(str).tolist()
    test_smiles = test_df["smiles_clean"].astype(str).tolist()

    y_train = train_df["binds"].astype(int).values
    ids_train = train_df["id"].values
    ids_test = test_df["id"].values

    print("\n[ecfp] Parameters:")
    print(f"  radius       = {radius}")
    print(f"  n_bits       = {n_bits}")
    print(f"  use_chirality= {use_chirality}")

    # Compute fingerprints
    print("\n[ecfp] Computing ECFP for train...")
    X_train = compute_ecfp_matrix(
        train_smiles,
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
        verbose=True,
    )

    print("[ecfp] Computing ECFP for test...")
    X_test = compute_ecfp_matrix(
        test_smiles,
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
        verbose=True,
    )

    # Save artifacts
    out_dir = Path(args.out_dir or PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[write] Saving feature matrices and arrays to:", out_dir)
    sparse.save_npz(out_dir / "X_train_full.npz", X_train)
    sparse.save_npz(out_dir / "X_test.npz", X_test)

    np.save(out_dir / "y_train_full.npy", y_train)
    np.save(out_dir / "ids_train_full.npy", ids_train)
    np.save(out_dir / "ids_test.npy", ids_test)

    # Metadata for reproducibility
    meta: Dict[str, Any] = {
        "radius": int(radius),
        "n_bits": int(n_bits),
        "use_chirality": bool(use_chirality),
        "train_source": str(train_path.resolve()),
        "test_source": str(test_path.resolve()),
        "outputs": {
            "X_train_full": str((out_dir / "X_train_full.npz").resolve()),
            "X_test": str((out_dir / "X_test.npz").resolve()),
            "y_train_full": str((out_dir / "y_train_full.npy").resolve()),
            "ids_train_full": str((out_dir / "ids_train_full.npy").resolve()),
            "ids_test": str((out_dir / "ids_test.npy").resolve()),
        },
    }
    joblib.dump(meta, out_dir / "prep_metadata.joblib")
    print("[write] prep_metadata.joblib")

    # Sanity checks
    run_sanity_checks(X_train, X_test, y_train, ids_train, ids_test)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Step 3: ECFP feature generation for BELKA–BRD4."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pecfp = sub.add_parser(
        "ecfp",
        help="Compute ECFP features from *_clean_blocks.parquet and save npz/npy/joblib.",
    )
    pecfp.add_argument(
        "--train-in",
        dest="train_in",
        default=None,
        help="Input train parquet (default: train_brd4_50k_clean_blocks.parquet)",
    )
    pecfp.add_argument(
        "--test-in",
        dest="test_in",
        default=None,
        help="Input test parquet (default: test_brd4_50k_clean_blocks.parquet)",
    )
    pecfp.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Output directory for npz/npy/joblib (default: data/processed)",
    )
    pecfp.add_argument(
        "--radius",
        type=int,
        default=2,
        help="ECFP radius (default: 2)",
    )
    pecfp.add_argument(
        "--n-bits",
        dest="n_bits",
        type=int,
        default=2048,
        help="Number of bits for ECFP (default: 2048)",
    )
    pecfp.add_argument(
        "--no-chirality",
        action="store_true",
        help="Disable useChirality in Morgan fingerprints (default: chirality ON).",
    )
    pecfp.set_defaults(func=run_ecfp)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
