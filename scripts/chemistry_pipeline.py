from __future__ import annotations

"""
A single, end-to-end pipeline that processes DEL chemistry data for modeling.
1) Clean DEL SMILES (replace [Dy] -> [H], canonicalize isomeric SMILES)
2) Canonicalize building block columns (bb*_smiles -> *_clean)
3) Mark train rows that share any block with test (shared_block_any)
4) Make an OOD validation split by holding out a percentage of blocks from TRAIN
5) Persist processed train/test parquet files and a metadata joblib

Defaults keep your current project layout:
  data/raw/train_brd4_50k_stratified.parquet
  data/raw/test_brd4_50k.parquet
  data/processed/{*_clean.parquet, *_clean_blocks.parquet, chemistry_blocks_metadata.joblib}

CLI examples:
  python scripts/chemistry_pipeline.py clean                # Step 1 only
  python scripts/chemistry_pipeline.py blocks               # Steps 2-5 (expects *_clean files exist)
  python scripts/chemistry_pipeline.py all                  # Run everything end-to-end
  
Options:
  --held-out-pct 0.05    (percentage of TRAIN blocks to hold out into val_ood)
  --seed 42              (deterministic seed for held-out selection)

This script is safe to run multiple times; it overwrites outputs.
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

# --------------------------------------------------------------------------------------
# Paths & utilities
# --------------------------------------------------------------------------------------

def get_project_root() -> Path:
    """Compute project root whether run as script or interactively."""
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        # __file__ not defined (e.g., in notebooks). Fall back to CWD.
        return Path.cwd()

PROJECT_ROOT = get_project_root()
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Default IO
TRAIN_IN_RAW = RAW_DIR / "train_brd4_50k_stratified.parquet"
TEST_IN_RAW  = RAW_DIR / "test_brd4_50k.parquet"

TRAIN_OUT_CLEAN = PROCESSED_DIR / "train_brd4_50k_clean.parquet"
TEST_OUT_CLEAN  = PROCESSED_DIR / "test_brd4_50k_clean.parquet"

TRAIN_OUT_BLOCKS = PROCESSED_DIR / "train_brd4_50k_clean_blocks.parquet"
TEST_OUT_BLOCKS  = PROCESSED_DIR / "test_brd4_50k_clean_blocks.parquet"
META_OUT         = PROCESSED_DIR / "chemistry_blocks_metadata.joblib"

BB_COLS = [
    "buildingblock1_smiles",
    "buildingblock2_smiles",
    "buildingblock3_smiles",
]

# --------------------------------------------------------------------------------------
# Core chemistry helpers
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
    if not isinstance(s, str):
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    Chem.SanitizeMol(mol, catchErrors=True)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# --------------------------------------------------------------------------------------
# Step 1: Clean DEL SMILES on train/test
# --------------------------------------------------------------------------------------

def process_smiles(df: pd.DataFrame) -> pd.DataFrame:
    if "molecule_smiles" not in df.columns:
        raise KeyError("Expected column 'molecule_smiles' in input DataFrame.")
    out = df.copy()
    out["smiles_clean"] = out["molecule_smiles"].astype(str).map(clean_smiles)
    return out


def step_clean(
    train_in: Path = TRAIN_IN_RAW,
    test_in: Path = TEST_IN_RAW,
    train_out: Path = TRAIN_OUT_CLEAN,
    test_out: Path = TEST_OUT_CLEAN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[read]  train: {train_in}")
    print(f"[read]  test : {test_in}")
    train = pd.read_parquet(train_in)
    test = pd.read_parquet(test_in)

    train_c = process_smiles(train)
    test_c = process_smiles(test)

    # audit
    n_train_bad = train_c["smiles_clean"].isna().sum()
    n_test_bad = test_c["smiles_clean"].isna().sum()
    print(f"[train] rows={len(train_c)}  invalid_smiles={n_train_bad}")
    print(f"[test ] rows={len(test_c)}   invalid_smiles={n_test_bad}")

    diffs = (
        train_c.assign(orig=train["molecule_smiles"])  # align original
        .loc[lambda d: d["smiles_clean"].ne(d["orig"])].head(5)
    )
    if not diffs.empty:
        print("\nExamples of changed SMILES (train):")
        print(diffs[["orig", "smiles_clean"]].to_string(index=False))
        print()

    # write
    train_c.to_parquet(train_out, index=False)
    test_c.to_parquet(test_out, index=False)
    print(f"[write] {train_out}")
    print(f"[write] {test_out}")
    return train_c, test_c


# --------------------------------------------------------------------------------------
# Steps 2-4: canonicalize blocks; shared flags; OOD split by held-out blocks
# --------------------------------------------------------------------------------------

def ensure_block_clean_cols(df: pd.DataFrame, bb_cols: Iterable[str] = BB_COLS) -> pd.DataFrame:
    out = df.copy()
    for c in bb_cols:
        cc = f"{c}_clean"
        if cc not in out.columns:
            out[cc] = out[c].astype(str).map(canon_block) if c in out.columns else None
    return out


def add_shared_block_flag(train: pd.DataFrame, test: pd.DataFrame, bb_cols: Iterable[str] = BB_COLS) -> pd.DataFrame:
    train = ensure_block_clean_cols(train, bb_cols)
    test = ensure_block_clean_cols(test, bb_cols)

    test_blocks: set[str] = set()
    for c in bb_cols:
        cc = f"{c}_clean"
        if cc in test.columns:
            test_blocks |= set(pd.Series(test[cc]).dropna().unique().tolist())

    def any_shared_blocks(row: pd.Series) -> bool:
        return any((row.get(f"{c}_clean") in test_blocks) for c in bb_cols)

    train = train.copy()
    train["shared_block_any"] = train.apply(any_shared_blocks, axis=1)
    return train


def make_ood_split(
    train: pd.DataFrame,
    held_out_pct: float = 0.05,
    seed: int = 42,
    bb_cols: Iterable[str] = BB_COLS,
) -> tuple[pd.DataFrame, list[str]]:
    train = ensure_block_clean_cols(train, bb_cols)
    bb_clean = [f"{c}_clean" for c in bb_cols]

    # Universe of unique blocks from TRAIN ONLY
    blocks_set = set().union(*[set(pd.Series(train[cc]).dropna().unique()) for cc in bb_clean])
    blocks = np.array(sorted(list(blocks_set)))

    if len(blocks) == 0:
        raise ValueError("No building blocks found to create an OOD split.")

    rng = np.random.default_rng(seed)
    k = max(1, int(round(held_out_pct * len(blocks))))
    held_out = set(rng.choice(blocks, size=k, replace=False))

    def any_held_out(row: pd.Series) -> bool:
        return any((row.get(cc) in held_out) for cc in bb_clean)

    train = train.copy()
    train["split_group"] = np.where(train.apply(any_held_out, axis=1), "val_ood", "train_in")
    return train, sorted(list(held_out))


# --------------------------------------------------------------------------------------
# Step 5: persist block-cleaned data + metadata
# --------------------------------------------------------------------------------------

def persist_blocks_and_meta(
    train: pd.DataFrame,
    test: pd.DataFrame,
    held_out_blocks: list[str],
    train_out: Path = TRAIN_OUT_BLOCKS,
    test_out: Path = TEST_OUT_BLOCKS,
    meta_out: Path = META_OUT,
    source_train: Path = TRAIN_OUT_CLEAN,
    source_test: Path = TEST_OUT_CLEAN,
    held_out_pct: float = 0.05,
    seed: int = 42,
) -> None:
    train.to_parquet(train_out, index=False)
    test.to_parquet(test_out, index=False)

    meta = {
        "bb_clean_cols": [f"{c}_clean" for c in BB_COLS],
        "split_col": "split_group",
        "held_out_blocks": list(held_out_blocks),
        "held_out_pct": float(held_out_pct),
        "rng_seed": int(seed),
        "source_train": str(source_train.resolve()),
        "source_test": str(source_test.resolve()),
        "outputs": {
            "train": str(Path(train_out).resolve()),
            "test": str(Path(test_out).resolve()),
        },
    }
    joblib.dump(meta, meta_out)

    print("[write]", Path(train_out).name)
    print("[write]", Path(test_out).name)
    print("[write]", Path(meta_out).name)

    # quick confirmation
    req_cols = set(["split_group"] + meta["bb_clean_cols"])
    print("\n[check] columns present in train:", req_cols.issubset(set(train.columns)))
    print("[check] split_group counts:\n", train["split_group"].value_counts())


# --------------------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------------------

def run_clean(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    return step_clean(
        train_in=Path(args.train_in or TRAIN_IN_RAW),
        test_in=Path(args.test_in or TEST_IN_RAW),
        train_out=Path(args.train_out or TRAIN_OUT_CLEAN),
        test_out=Path(args.test_out or TEST_OUT_CLEAN),
    )


def run_blocks(args: argparse.Namespace) -> None:
    # read cleaned files (can also accept overrides)
    train = pd.read_parquet(Path(args.train_in or TRAIN_OUT_CLEAN))
    test = pd.read_parquet(Path(args.test_in or TEST_OUT_CLEAN))

    # 2) ensure *_clean block columns
    train = ensure_block_clean_cols(train)
    test = ensure_block_clean_cols(test)

    # 3) shared flag
    train = add_shared_block_flag(train, test)

    # 4) OOD split
    train, held_out = make_ood_split(train, held_out_pct=args.held_out_pct, seed=args.seed)

    # 5) persist
    persist_blocks_and_meta(
        train,
        test,
        held_out,
        train_out=Path(args.train_out or TRAIN_OUT_BLOCKS),
        test_out=Path(args.test_out or TEST_OUT_BLOCKS),
        meta_out=Path(args.meta_out or META_OUT),
        source_train=Path(args.source_train or TRAIN_OUT_CLEAN),
        source_test=Path(args.source_test or TEST_OUT_CLEAN),
        held_out_pct=args.held_out_pct,
        seed=args.seed,
    )


def run_all(args: argparse.Namespace) -> None:
    # Step 1
    train_c, test_c = step_clean(
        train_in=Path(args.raw_train_in or TRAIN_IN_RAW),
        test_in=Path(args.raw_test_in or TEST_IN_RAW),
        train_out=Path(args.clean_train_out or TRAIN_OUT_CLEAN),
        test_out=Path(args.clean_test_out or TEST_OUT_CLEAN),
    )

    # Steps 2-5
    train_c = ensure_block_clean_cols(train_c)
    test_c = ensure_block_clean_cols(test_c)
    train_c = add_shared_block_flag(train_c, test_c)
    train_c, held_out = make_ood_split(train_c, held_out_pct=args.held_out_pct, seed=args.seed)

    persist_blocks_and_meta(
        train_c,
        test_c,
        held_out,
        train_out=Path(args.blocks_train_out or TRAIN_OUT_BLOCKS),
        test_out=Path(args.blocks_test_out or TEST_OUT_BLOCKS),
        meta_out=Path(args.meta_out or META_OUT),
        source_train=Path(args.clean_train_out or TRAIN_OUT_CLEAN),
        source_test=Path(args.clean_test_out or TEST_OUT_CLEAN),
        held_out_pct=args.held_out_pct,
        seed=args.seed,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chemistry preprocessing + block-based OOD pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # clean
    pc = sub.add_parser("clean", help="Clean DEL SMILES and write *_clean.parquet files")
    pc.add_argument("--train-in", dest="train_in", default=None, help="Path to raw train parquet")
    pc.add_argument("--test-in", dest="test_in", default=None, help="Path to raw test parquet")
    pc.add_argument("--train-out", dest="train_out", default=None, help="Output cleaned train parquet")
    pc.add_argument("--test-out", dest="test_out", default=None, help="Output cleaned test parquet")
    pc.set_defaults(func=run_clean)

    # blocks
    pb = sub.add_parser("blocks", help="Canonicalize block cols, flag shared, create OOD split, persist + metadata")
    pb.add_argument("--train-in", dest="train_in", default=None, help="Input cleaned train parquet")
    pb.add_argument("--test-in", dest="test_in", default=None, help="Input cleaned test parquet")
    pb.add_argument("--train-out", dest="train_out", default=None, help="Output blocks train parquet")
    pb.add_argument("--test-out", dest="test_out", default=None, help="Output blocks test parquet")
    pb.add_argument("--meta-out", dest="meta_out", default=None, help="Output joblib metadata path")
    pb.add_argument("--source-train", dest="source_train", default=None, help="Path recorded in metadata: cleaned train source")
    pb.add_argument("--source-test", dest="source_test", default=None, help="Path recorded in metadata: cleaned test source")
    pb.add_argument("--held-out-pct", type=float, default=0.05, help="Pct of TRAIN blocks held-out to val_ood [0-1]")
    pb.add_argument("--seed", type=int, default=42, help="Random seed for held-out selection")
    pb.set_defaults(func=run_blocks)

    # all
    pa = sub.add_parser("all", help="Run clean + blocks in sequence and write all outputs")
    # raw inputs for step 1
    pa.add_argument("--raw-train-in", dest="raw_train_in", default=None, help="Raw train parquet path")
    pa.add_argument("--raw-test-in", dest="raw_test_in", default=None, help="Raw test parquet path")
    # cleaned outputs (and sources recorded by metadata)
    pa.add_argument("--clean-train-out", dest="clean_train_out", default=None, help="Output cleaned train parquet")
    pa.add_argument("--clean-test-out", dest="clean_test_out", default=None, help="Output cleaned test parquet")
    # final blocks outputs
    pa.add_argument("--blocks-train-out", dest="blocks_train_out", default=None, help="Output blocks train parquet")
    pa.add_argument("--blocks-test-out", dest="blocks_test_out", default=None, help="Output blocks test parquet")
    pa.add_argument("--meta-out", dest="meta_out", default=None, help="Output joblib metadata path")
    pa.add_argument("--held-out-pct", type=float, default=0.05, help="Pct of TRAIN blocks held-out to val_ood [0-1]")
    pa.add_argument("--seed", type=int, default=42, help="Random seed for held-out selection")
    pa.set_defaults(func=run_all)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

