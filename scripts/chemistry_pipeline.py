from __future__ import annotations

"""
End-to-end pipeline to build an internal OOD benchmark from Kaggle's BRD4 TRAIN only.

Pipeline:
  1) Clean DEL SMILES (replace [Dy] -> [H], canonicalize isomeric SMILES).
  2) Canonicalize building block columns (bb*_smiles -> *_clean).
  3) Partition the universe of unique building blocks into disjoint sets:
       - TRAIN_BLOCKS
       - VAL_BLOCKS (held-out for validation OOD)
       - TEST_BLOCKS (held-out for test OOD)
     Assignment precedence per molecule: test_ood > val_ood > train_in.
  4) Persist:
       - One parquet with all rows and split_group column.
       - Separate per-split parquets for convenience.
       - A metadata joblib with block sets and parameters.

Defaults keep your current project layout:
  data/raw/train_brd4_50k_stratified.parquet
  data/processed/train_brd4_50k_clean.parquet
  data/processed/train_brd4_50k_clean_blocks.parquet
  data/processed/train_brd4_50k_clean_blocks_{train,val,test}.parquet
  data/processed/chemistry_blocks_metadata.joblib

CLI examples:
  # Step 1 only: clean SMILES and write *_clean.parquet
  python scripts/chemistry_pipeline.py clean

  # Step 2-4 only: assumes *_clean.parquet exists
  python scripts/chemistry_pipeline.py split --val-pct 0.08 --test-pct 0.08

  # All steps in sequence
  python scripts/chemistry_pipeline.py all --val-pct 0.08 --test-pct 0.08 --seed 42

Notes:
  - val_pct + test_pct must be in (0,1), and val_pct + test_pct < 1.
  - OOD is defined at block-level: if a molecule contains any held-out block,
    it belongs to the corresponding OOD split (test first, then val).
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, Set, Dict

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------

def get_project_root() -> Path:
    """Compute project root whether run as script or interactively."""
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()

PROJECT_ROOT = get_project_root()
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_TRAIN_IN   = RAW_DIR / "train_brd4_50k_stratified.parquet"

CLEAN_OUT      = PROCESSED_DIR / "train_brd4_50k_clean.parquet"
BLOCKS_ALL_OUT = PROCESSED_DIR / "train_brd4_50k_clean_blocks.parquet"
BLOCKS_TR_OUT  = PROCESSED_DIR / "train_brd4_50k_clean_blocks_train.parquet"
BLOCKS_VA_OUT  = PROCESSED_DIR / "train_brd4_50k_clean_blocks_val.parquet"
BLOCKS_TE_OUT  = PROCESSED_DIR / "train_brd4_50k_clean_blocks_test.parquet"
META_OUT       = PROCESSED_DIR / "chemistry_blocks_metadata.joblib"

# Building block columns in the raw file
BB_COLS = [
    "buildingblock1_smiles",
    "buildingblock2_smiles",
    "buildingblock3_smiles",
]

# --------------------------------------------------------------------------------------
# Chemistry helpers
# --------------------------------------------------------------------------------------

def clean_smiles(s: str) -> Optional[str]:
    """Replace DEL linker [Dy] with [H] and return canonical isomeric SMILES; None if invalid."""
    if s is None:
        return None
    s2 = str(s).replace("[Dy]", "[H]")
    mol = Chem.MolFromSmiles(s2)
    if mol is None:
        return None
    Chem.SanitizeMol(mol, catchErrors=True)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def canon_block(s: str) -> Optional[str]:
    """Canonicalize a block SMILES to isomeric SMILES; None if invalid."""
    if not isinstance(s, str):
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    Chem.SanitizeMol(mol, catchErrors=True)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def ensure_block_clean_cols(df: pd.DataFrame, bb_cols: Iterable[str] = BB_COLS) -> pd.DataFrame:
    """Make sure *_clean block columns exist; fill them by canonicalizing the originals."""
    out = df.copy()
    for c in bb_cols:
        cc = f"{c}_clean"
        if cc not in out.columns:
            if c not in out.columns:
                out[cc] = None
            else:
                out[cc] = out[c].astype(str).map(canon_block)
    return out


def process_smiles_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create 'smiles_clean' from 'molecule_smiles'."""
    if "molecule_smiles" not in df.columns:
        raise KeyError("Expected column 'molecule_smiles' in input DataFrame.")
    out = df.copy()
    out["smiles_clean"] = out["molecule_smiles"].astype(str).map(clean_smiles)
    return out

# --------------------------------------------------------------------------------------
# OOD split logic
# --------------------------------------------------------------------------------------

def collect_unique_blocks(train: pd.DataFrame, bb_cols: Iterable[str]) -> np.ndarray:
    """Collect unique non-null cleaned blocks from TRAIN."""
    bb_clean = [f"{c}_clean" for c in bb_cols]
    blocks_set = set().union(*[set(pd.Series(train[cc]).dropna().unique()) for cc in bb_clean])
    blocks = np.array(sorted(list(blocks_set)))
    if len(blocks) == 0:
        raise ValueError("No building blocks found in TRAIN to create an OOD split.")
    return blocks


def sample_block_partitions(
    blocks: np.ndarray,
    val_pct: float,
    test_pct: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Partition block universe into train/val/test sets by sampling disjoint val and test subsets.
    Remaining blocks are assigned to train.
    """
    if not (0 < val_pct < 1) or not (0 < test_pct < 1) or not (val_pct + test_pct < 1):
        raise ValueError("Require 0 < val_pct, test_pct and val_pct + test_pct < 1.")
    rng = np.random.default_rng(seed)

    n = len(blocks)
    n_val = max(1, int(round(val_pct * n)))
    n_test = max(1, int(round(test_pct * n)))

    # sample VAL
    val_idx = rng.choice(n, size=n_val, replace=False)
    val_blocks = set(blocks[val_idx])

    # sample TEST from the remaining
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[val_idx] = False
    remaining = blocks[remaining_mask]
    if len(remaining) < n_test:
        raise ValueError("Not enough blocks remaining to sample TEST set. Reduce test_pct or val_pct.")
    test_idx_local = rng.choice(len(remaining), size=n_test, replace=False)
    test_blocks = set(remaining[test_idx_local])

    train_blocks = set(blocks) - val_blocks - test_blocks
    return train_blocks, val_blocks, test_blocks


def assign_split_groups(
    df: pd.DataFrame,
    val_blocks: Set[str],
    test_blocks: Set[str],
    bb_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Vectorized split assignment with precedence: test_ood > val_ood > train_in.
    A molecule goes to OOD if ANY of its blocks is in the held-out set.
    """
    bb_clean = [f"{c}_clean" for c in bb_cols]
    out = df.copy()

    # Boolean masks: does the row contain any val/test block?
    is_val_any = None
    is_test_any = None
    for cc in bb_clean:
        col = out[cc]
        val_hit = col.isin(val_blocks)
        test_hit = col.isin(test_blocks)
        is_val_any = val_hit if is_val_any is None else (is_val_any | val_hit)
        is_test_any = test_hit if is_test_any is None else (is_test_any | test_hit)

    out["split_group"] = "train_in"
    # Assign val first, then test to ensure test overrides val
    out.loc[is_val_any, "split_group"] = "val_ood"
    out.loc[is_test_any, "split_group"] = "test_ood"
    return out


def sanity_checks_split(
    df: pd.DataFrame,
    train_blocks: Set[str],
    val_blocks: Set[str],
    test_blocks: Set[str],
    bb_cols: Iterable[str],
) -> Dict[str, int]:
    """
    Ensure no held-out blocks leak into train_in.
    Return row counts per split for convenience.
    """
    bb_clean = [f"{c}_clean" for c in bb_cols]
    # Blocks actually observed in each split
    blocks_in_train = set()
    blocks_in_val   = set()
    blocks_in_test  = set()
    for cc in bb_clean:
        blocks_in_train |= set(pd.Series(df.loc[df["split_group"] == "train_in", cc]).dropna().unique())
        blocks_in_val   |= set(pd.Series(df.loc[df["split_group"] == "val_ood",   cc]).dropna().unique())
        blocks_in_test  |= set(pd.Series(df.loc[df["split_group"] == "test_ood",  cc]).dropna().unique())

    # No held-out blocks in train_in
    leak_val = blocks_in_train & val_blocks
    leak_test = blocks_in_train & test_blocks
    if leak_val or leak_test:
        raise AssertionError(
            f"Leakage detected: train_in contains held-out blocks. "
            f"val_leaks={len(leak_val)}, test_leaks={len(leak_test)}"
        )

    counts = df["split_group"].value_counts().to_dict()
    counts.setdefault("train_in", 0)
    counts.setdefault("val_ood", 0)
    counts.setdefault("test_ood", 0)
    return counts

# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------

def step_clean(
    raw_train_in: Path = RAW_TRAIN_IN,
    clean_out: Path = CLEAN_OUT,
) -> pd.DataFrame:
    print(f"[read]  raw train: {raw_train_in}")
    train = pd.read_parquet(raw_train_in)

    train_c = process_smiles_column(train)
    n_bad = train_c["smiles_clean"].isna().sum()
    print(f"[clean] rows={len(train_c)}  invalid_smiles={n_bad}")

    diffs = (
        train_c.assign(orig=train["molecule_smiles"])
        .loc[lambda d: d["smiles_clean"].ne(d["orig"])].head(5)
    )
    if not diffs.empty:
        print("\nExamples of changed SMILES (train):")
        print(diffs[["orig", "smiles_clean"]].to_string(index=False))
        print()

    train_c.to_parquet(clean_out, index=False)
    print(f"[write] {clean_out}")
    return train_c


def step_split(
    cleaned_in: Path = CLEAN_OUT,
    all_out: Path = BLOCKS_ALL_OUT,
    train_out: Path = BLOCKS_TR_OUT,
    val_out: Path = BLOCKS_VA_OUT,
    test_out: Path = BLOCKS_TE_OUT,
    meta_out: Path = META_OUT,
    val_pct: float = 0.08,
    test_pct: float = 0.08,
    seed: int = 42,
) -> None:
    print(f"[read] cleaned: {cleaned_in}")
    df = pd.read_parquet(cleaned_in)

    # Ensure *_clean block columns exist
    df = ensure_block_clean_cols(df, BB_COLS)

    # Build block partitions
    blocks = collect_unique_blocks(df, BB_COLS)
    print(f"[blocks] unique cleaned blocks in TRAIN: {len(blocks)}")

    train_blocks, val_blocks, test_blocks = sample_block_partitions(
        blocks=blocks, val_pct=val_pct, test_pct=test_pct, seed=seed
    )
    print(f"[partition] #train_blocks={len(train_blocks)} "
          f"#val_blocks={len(val_blocks)} #test_blocks={len(test_blocks)}")

    # Assign split groups; precedence test > val > train
    df_split = assign_split_groups(df, val_blocks, test_blocks, BB_COLS)

    # Sanity checks + counts
    counts = sanity_checks_split(df_split, train_blocks, val_blocks, test_blocks, BB_COLS)
    print("[counts] train_in={train_in}  val_ood={val_ood}  test_ood={test_ood}".format(**counts))

    # Persist all rows + per-split convenience files
    df_split.to_parquet(all_out, index=False)
    df_split.loc[df_split["split_group"] == "train_in"].to_parquet(train_out, index=False)
    df_split.loc[df_split["split_group"] == "val_ood"].to_parquet(val_out, index=False)
    df_split.loc[df_split["split_group"] == "test_ood"].to_parquet(test_out, index=False)

    # Metadata
    meta = {
        "bb_clean_cols": [f"{c}_clean" for c in BB_COLS],
        "split_col": "split_group",
        "seed": int(seed),
        "val_pct": float(val_pct),
        "test_pct": float(test_pct),
        "block_partitions": {
            "train_blocks": sorted(list(train_blocks)),
            "val_blocks": sorted(list(val_blocks)),
            "test_blocks": sorted(list(test_blocks)),
        },
        "source_cleaned": str(Path(cleaned_in).resolve()),
        "outputs": {
            "all_rows": str(Path(all_out).resolve()),
            "train": str(Path(train_out).resolve()),
            "val": str(Path(val_out).resolve()),
            "test": str(Path(test_out).resolve()),
        },
    }
    joblib.dump(meta, meta_out)

    print(f"[write] {all_out.name}")
    print(f"[write] {train_out.name}")
    print(f"[write] {val_out.name}")
    print(f"[write] {test_out.name}")
    print(f"[write] {meta_out.name}")

# --------------------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------------------

def run_clean(args: argparse.Namespace) -> None:
    step_clean(
        raw_train_in=Path(args.raw_train_in or RAW_TRAIN_IN),
        clean_out=Path(args.clean_out or CLEAN_OUT),
    )


def run_split(args: argparse.Namespace) -> None:
    step_split(
        cleaned_in=Path(args.cleaned_in or CLEAN_OUT),
        all_out=Path(args.all_out or BLOCKS_ALL_OUT),
        train_out=Path(args.train_out or BLOCKS_TR_OUT),
        val_out=Path(args.val_out or BLOCKS_VA_OUT),
        test_out=Path(args.test_out or BLOCKS_TE_OUT),
        meta_out=Path(args.meta_out or META_OUT),
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,
    )


def run_all(args: argparse.Namespace) -> None:
    # Step 1
    step_clean(
        raw_train_in=Path(args.raw_train_in or RAW_TRAIN_IN),
        clean_out=Path(args.clean_out or CLEAN_OUT),
    )
    # Step 2-4
    step_split(
        cleaned_in=Path(args.cleaned_in or args.clean_out or CLEAN_OUT),
        all_out=Path(args.all_out or BLOCKS_ALL_OUT),
        train_out=Path(args.train_out or BLOCKS_TR_OUT),
        val_out=Path(args.val_out or BLOCKS_VA_OUT),
        test_out=Path(args.test_out or BLOCKS_TE_OUT),
        meta_out=Path(args.meta_out or META_OUT),
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,
    )

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chemistry preprocessing + block-based OOD split (train-only)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # clean
    pc = sub.add_parser("clean", help="Clean SMILES from Kaggle train and write *_clean.parquet")
    pc.add_argument("--raw-train-in", dest="raw_train_in", default=None, help="Raw Kaggle train parquet path")
    pc.add_argument("--clean-out", dest="clean_out", default=None, help="Output cleaned train parquet")
    pc.set_defaults(func=run_clean)

    # split
    ps = sub.add_parser("split", help="Create OOD split by holding out blocks into val/test; write blocks+metadata")
    ps.add_argument("--cleaned-in", dest="cleaned_in", default=None, help="Input cleaned train parquet")
    ps.add_argument("--all-out", dest="all_out", default=None, help="Output parquet with all rows + split_group")
    ps.add_argument("--train-out", dest="train_out", default=None, help="Output parquet for train_in")
    ps.add_argument("--val-out", dest="val_out", default=None, help="Output parquet for val_ood")
    ps.add_argument("--test-out", dest="test_out", default=None, help="Output parquet for test_ood")
    ps.add_argument("--meta-out", dest="meta_out", default=None, help="Output joblib metadata path")
    ps.add_argument("--val-pct", type=float, default=0.08, help="Fraction of blocks held out to VAL OOD (0,1)")
    ps.add_argument("--test-pct", type=float, default=0.08, help="Fraction of blocks held out to TEST OOD (0,1)")
    ps.add_argument("--seed", type=int, default=42, help="Random seed for block partitioning")
    ps.set_defaults(func=run_split)

    # all
    pa = sub.add_parser("all", help="Run clean + split in sequence and write all outputs")
    pa.add_argument("--raw-train-in", dest="raw_train_in", default=None, help="Raw Kaggle train parquet path")
    pa.add_argument("--clean-out", dest="clean_out", default=None, help="Output cleaned train parquet")
    pa.add_argument("--cleaned-in", dest="cleaned_in", default=None, help="Input cleaned train parquet (if skipping clean)")
    pa.add_argument("--all-out", dest="all_out", default=None, help="Output parquet with all rows + split_group")
    pa.add_argument("--train-out", dest="train_out", default=None, help="Output parquet for train_in")
    pa.add_argument("--val-out", dest="val_out", default=None, help="Output parquet for val_ood")
    pa.add_argument("--test-out", dest="test_out", default=None, help="Output parquet for test_ood")
    pa.add_argument("--meta-out", dest="meta_out", default=None, help="Output joblib metadata path")
    pa.add_argument("--val-pct", type=float, default=0.08, help="Fraction of blocks held out to VAL OOD (0,1)")
    pa.add_argument("--test-pct", type=float, default=0.08, help="Fraction of blocks held out to TEST OOD (0,1)")
    pa.add_argument("--seed", type=int, default=42, help="Random seed for block partitioning")
    pa.set_defaults(func=run_all)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


