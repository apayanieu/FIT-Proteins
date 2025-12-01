#!/usr/bin/env python
"""
SMILES preprocessing for neural networks on BRD4 (BELKA 50k).

Reads the *chemistry_pipeline* outputs:

  data/processed/train_brd4_50k_clean_blocks_train.parquet
  data/processed/train_brd4_50k_clean_blocks_val.parquet
  data/processed/train_brd4_50k_clean_blocks_test.parquet

and produces NN-ready tokenized SMILES arrays:

  data/processed/smiles_cnn_train_in.npz
  data/processed/smiles_cnn_val_ood.npz
  data/processed/smiles_cnn_test_ood.npz
  data/processed/smiles_cnn_metadata.joblib

Tokenization:
  - Simple SMILES tokenizer (handles "Cl", "Br" as 2-char tokens).
  - Vocabulary learned on TRAIN ONLY.
  - PAD = 0, UNK = 1, all other tokens start from 2.
  - Sequences are truncated / padded to max_len (default 256).

CLI:
  python scripts/smiles_nn_preproc.py prepare \
      --label-col binds \
      --max-len 256
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Paths & constants (mirroring chemistry_pipeline.py)
# --------------------------------------------------------------------------------------

def get_project_root() -> Path:
    """Compute project root whether run as script or interactively."""
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        # Fallback if __file__ is not defined (interactive)
        return Path.cwd()


PROJECT_ROOT = get_project_root()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Chemistry-pipeline outputs
BLOCKS_TR_OUT = PROCESSED_DIR / "train_brd4_50k_clean_blocks_train.parquet"
BLOCKS_VA_OUT = PROCESSED_DIR / "train_brd4_50k_clean_blocks_val.parquet"
BLOCKS_TE_OUT = PROCESSED_DIR / "train_brd4_50k_clean_blocks_test.parquet"

# Outputs for NN
SMILES_TRAIN_OUT = PROCESSED_DIR / "smiles_cnn_train_in.npz"
SMILES_VAL_OUT   = PROCESSED_DIR / "smiles_cnn_val_ood.npz"
SMILES_TEST_OUT  = PROCESSED_DIR / "smiles_cnn_test_ood.npz"
META_OUT         = PROCESSED_DIR / "smiles_cnn_metadata.joblib"


# --------------------------------------------------------------------------------------
# Tokenization helpers
# --------------------------------------------------------------------------------------

def tokenize_smiles(smiles: str) -> List[str]:
    """
    Very simple SMILES tokenizer.
    - Treats "Cl" and "Br" as 2-char tokens.
    - Everything else is single-char.
    """
    tokens: List[str] = []
    i = 0
    L = len(smiles)
    while i < L:
        ch = smiles[i]
        # Two-char atoms
        if i < L - 1:
            ch2 = smiles[i:i+2]
            if ch2 in ("Cl", "Br"):
                tokens.append(ch2)
                i += 2
                continue
        tokens.append(ch)
        i += 1
    return tokens


def build_vocab(smiles_list: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """
    Build token→index vocab from a list of SMILES strings (TRAIN ONLY).
    Reserve:
      0 -> PAD
      1 -> UNK
    """
    token_set = set()
    for s in smiles_list:
        for tok in tokenize_smiles(s):
            token_set.add(tok)

    tokens_sorted = sorted(token_set)
    token2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, tok in enumerate(tokens_sorted, start=2):
        token2idx[tok] = i

    idx2token = ["<PAD>", "<UNK>"] + tokens_sorted
    return token2idx, idx2token


def smiles_to_ids(
    smiles: str,
    token2idx: Dict[str, int],
    max_len: int,
) -> np.ndarray:
    """
    Convert a SMILES string to a fixed-length sequence of token IDs.
    """
    tokens = tokenize_smiles(smiles)
    ids = [token2idx.get(tok, token2idx["<UNK>"]) for tok in tokens[:max_len]]
    # pad
    if len(ids) < max_len:
        ids += [token2idx["<PAD>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def preprocess_split(
    parquet_path: Path,
    token2idx: Dict[str, int],
    max_len: int,
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a split parquet, keep rows with non-null smiles_clean & labels,
    return (X, y, ids).
    """
    df = pd.read_parquet(parquet_path)
    if "smiles_clean" not in df.columns:
        raise KeyError(f"'smiles_clean' not found in {parquet_path}")

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {parquet_path}")

    mask = df["smiles_clean"].notna() & df[label_col].notna()
    df = df.loc[mask].copy()

    smiles_list = df["smiles_clean"].astype(str).tolist()
    y = df[label_col].astype(int).to_numpy()
    ids = df["molecule_id"].astype(str).to_numpy() if "molecule_id" in df.columns else np.arange(len(df))

    X = np.stack(
        [smiles_to_ids(s, token2idx=token2idx, max_len=max_len) for s in smiles_list],
        axis=0,
    )

    return X, y, ids


# --------------------------------------------------------------------------------------
# Main orchestrator
# --------------------------------------------------------------------------------------

def run_prepare(args: argparse.Namespace) -> None:
    label_col: str = args.label_col
    max_len: int = args.max_len

    print("=== SMILES NN PREPROCESSING ===")
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Processed dir : {PROCESSED_DIR}")
    print(f"Label column  : {label_col}")
    print(f"Max length    : {max_len}")

    # --- 1) Read TRAIN split to build vocab ---
    print(f"[read] TRAIN from {BLOCKS_TR_OUT}")
    df_tr = pd.read_parquet(BLOCKS_TR_OUT)

    if "smiles_clean" not in df_tr.columns:
        raise KeyError("Expected column 'smiles_clean' in TRAIN parquet.")

    if label_col not in df_tr.columns:
        raise KeyError(f"Expected label column '{label_col}' in TRAIN parquet.")

    mask_tr = df_tr["smiles_clean"].notna() & df_tr[label_col].notna()
    df_tr = df_tr.loc[mask_tr].copy()

    smiles_train = df_tr["smiles_clean"].astype(str).tolist()
    y_train = df_tr[label_col].astype(int).to_numpy()

    print(f"[train] rows={len(df_tr)}  positives={int(y_train.sum())}  "
          f"negatives={len(y_train) - int(y_train.sum())}  "
          f"pos_rate={y_train.mean():.6f}")

    # Build vocab on TRAIN only
    token2idx, idx2token = build_vocab(smiles_train)
    pad_idx = token2idx["<PAD>"]
    unk_idx = token2idx["<UNK>"]

    # Inspect lengths
    lengths = np.array([len(tokenize_smiles(s)) for s in smiles_train], dtype=np.int32)
    max_seen = int(lengths.max())
    p95 = float(np.percentile(lengths, 95))
    print(f"[lengths] max_seen={max_seen}  p95={p95:.1f}")

    if max_len > max_seen:
        # No need to pad to something bigger than necessary
        max_len = max_seen
        print(f"[lengths] adjusting max_len down to {max_len}")

    # --- 2) Process all three splits ---
    print(f"[prep] TRAIN_IN from {BLOCKS_TR_OUT}")
    X_tr, y_tr, ids_tr = preprocess_split(
        parquet_path=BLOCKS_TR_OUT,
        token2idx=token2idx,
        max_len=max_len,
        label_col=label_col,
    )

    print(f"[prep] VAL_OOD from {BLOCKS_VA_OUT}")
    X_va, y_va, ids_va = preprocess_split(
        parquet_path=BLOCKS_VA_OUT,
        token2idx=token2idx,
        max_len=max_len,
        label_col=label_col,
    )

    print(f"[prep] TEST_OOD from {BLOCKS_TE_OUT}")
    X_te, y_te, ids_te = preprocess_split(
        parquet_path=BLOCKS_TE_OUT,
        token2idx=token2idx,
        max_len=max_len,
        label_col=label_col,
    )

    # --- 3) Save npz files ---
    print(f"[write] {SMILES_TRAIN_OUT}")
    np.savez_compressed(SMILES_TRAIN_OUT, X=X_tr, y=y_tr, ids=ids_tr)

    print(f"[write] {SMILES_VAL_OUT}")
    np.savez_compressed(SMILES_VAL_OUT, X=X_va, y=y_va, ids=ids_va)

    print(f"[write] {SMILES_TEST_OUT}")
    np.savez_compressed(SMILES_TEST_OUT, X=X_te, y=y_te, ids=ids_te)

    # --- 4) Save metadata ---
    meta = {
        "token2idx": token2idx,
        "idx2token": idx2token,
        "pad_idx": pad_idx,
        "unk_idx": unk_idx,
        "max_len": max_len,
        "label_col": label_col,
        "source_parquets": {
            "train_in": str(BLOCKS_TR_OUT.resolve()),
            "val_ood": str(BLOCKS_VA_OUT.resolve()),
            "test_ood": str(BLOCKS_TE_OUT.resolve()),
        },
        "stats": {
            "train_pos_rate": float(y_tr.mean()),
            "val_pos_rate": float(y_va.mean()),
            "test_pos_rate": float(y_te.mean()),
        },
    }

    joblib.dump(meta, META_OUT)
    print(f"[write] metadata -> {META_OUT}")
    print("=== DONE SMILES NN PREPROCESSING ===")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess SMILES for neural networks (BELKA BRD4)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="Tokenize SMILES & write NN-ready arrays")
    pp.add_argument("--label-col", type=str, default="binds",
                    help="Name of binary label column in the parquet files (default: 'binds')")
    pp.add_argument("--max-len", type=int, default=256,
                    help="Max SMILES token length (truncate/pad to this length; default=256)")
    pp.set_defaults(func=run_prepare)

    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
