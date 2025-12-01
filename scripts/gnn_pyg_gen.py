#!/usr/bin/env python
"""
GNN preprocessing for BRD4 binding (BELKA 50k) using PyTorch Geometric.

Reads the *chemistry_pipeline* outputs:

  data/processed/train_brd4_50k_clean_blocks_train.parquet
  data/processed/train_brd4_50k_clean_blocks_val.parquet
  data/processed/train_brd4_50k_clean_blocks_test.parquet

and converts each molecule into a PyG `Data` graph with:

  - node features: atom type one-hot + formal charge + aromatic flag
  - edge features: bond type one-hot + conjugation flag + ring flag

Outputs:

  data/processed/gnn_pyg_train_in.pt
  data/processed/gnn_pyg_val_ood.pt
  data/processed/gnn_pyg_test_ood.pt
  data/processed/gnn_pyg_metadata.joblib

Each *.pt file is a list[torch_geometric.data.Data] with `.x`, `.edge_index`,
`.edge_attr`, `.y`.

CLI:

  python scripts/gnn_pyg_preproc.py prepare --label-col binds
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm


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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Chemistry pipeline outputs (already split by building blocks)
BLOCKS_TR_OUT = PROCESSED_DIR / "train_brd4_50k_clean_blocks_train.parquet"
BLOCKS_VA_OUT = PROCESSED_DIR / "train_brd4_50k_clean_blocks_val.parquet"
BLOCKS_TE_OUT = PROCESSED_DIR / "train_brd4_50k_clean_blocks_test.parquet"

# Outputs for GNN
GNN_TRAIN_OUT = PROCESSED_DIR / "gnn_pyg_train_in.pt"
GNN_VAL_OUT   = PROCESSED_DIR / "gnn_pyg_val_ood.pt"
GNN_TEST_OUT  = PROCESSED_DIR / "gnn_pyg_test_ood.pt"
META_OUT      = PROCESSED_DIR / "gnn_pyg_metadata.joblib"


# --------------------------------------------------------------------------------------
# Featurization helpers (mirroring the notebook)
# --------------------------------------------------------------------------------------

ATOM_TYPES = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot_with_other(x, choices):
    """One-hot encode x over `choices`, with an extra 'other' bucket."""
    v = [0] * (len(choices) + 1)
    if x in choices:
        v[choices.index(x)] = 1
    else:
        v[-1] = 1
    return v


def atom_to_feature_vector(atom: Chem.rdchem.Atom) -> torch.Tensor:
    atom_type = one_hot_with_other(atom.GetSymbol(), ATOM_TYPES)
    formal_charge = atom.GetFormalCharge()
    is_aromatic = int(atom.GetIsAromatic())
    return torch.tensor(atom_type + [formal_charge, is_aromatic], dtype=torch.float)


def bond_to_feature_vector(bond: Chem.rdchem.Bond) -> torch.Tensor:
    bt = bond.GetBondType()
    bond_type_oh = one_hot_with_other(bt, BOND_TYPES)
    is_conjugated = int(bond.GetIsConjugated())
    is_in_ring = int(bond.IsInRing())
    return torch.tensor(bond_type_oh + [is_conjugated, is_in_ring], dtype=torch.float)


NODE_FEAT_DIM = len(ATOM_TYPES) + 1 + 1 + 1   # atom types + other + charge + aromatic
EDGE_FEAT_DIM = len(BOND_TYPES) + 1 + 1 + 1   # bond types + other + conjugated + in_ring


def smiles_to_data(smiles: str, y_value=None) -> Optional[Data]:
    """Convert a SMILES string into a PyG Data graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    Chem.SanitizeMol(mol)

    # node features
    x_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    if len(x_list) == 0:
        return None
    x = torch.stack(x_list, dim=0)  # [num_nodes, NODE_FEAT_DIM]

    # edges (undirected; we store both directions)
    edge_index_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = bond_to_feature_vector(bond)

        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(e)
        edge_attr_list.append(e)

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEAT_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_list, dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y_value is not None:
        data.y = torch.tensor([float(y_value)], dtype=torch.float)
    return data


# --------------------------------------------------------------------------------------
# Core preprocessing
# --------------------------------------------------------------------------------------

def build_graphs_from_parquet(
    parquet_path: Path,
    smiles_col: str,
    label_col: Optional[str],
) -> Tuple[List[Data], Dict[str, float]]:
    """
    Build a list of PyG Data graphs from a parquet split.
    If label_col is None, graphs will not have .y.
    Returns (graphs, stats).
    """
    df = pd.read_parquet(parquet_path)

    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found in {parquet_path}")

    if label_col is not None and label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {parquet_path}")

    if label_col is None:
        iterator = zip(df[smiles_col], [None] * len(df))
    else:
        iterator = zip(df[smiles_col], df[label_col])

    graphs: List[Data] = []
    n_rows = 0
    n_valid = 0
    n_pos = 0

    for smiles, y in tqdm(iterator, total=len(df), desc=f"Graphs from {parquet_path.name}"):
        n_rows += 1
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
            continue
        if label_col is not None and (y is None or (isinstance(y, float) and np.isnan(y))):
            continue

        g = smiles_to_data(str(smiles), y if label_col is not None else None)
        if g is None:
            continue

        graphs.append(g)
        n_valid += 1
        if label_col is not None and float(y) == 1.0:
            n_pos += 1

    stats = {
        "n_rows": float(n_rows),
        "n_graphs": float(n_valid),
        "n_pos": float(n_pos),
        "pos_rate": float(n_pos) / float(n_valid) if (n_valid > 0 and label_col is not None) else float("nan"),
    }
    return graphs, stats


def run_prepare(args: argparse.Namespace) -> None:
    smiles_col: str = args.smiles_col
    label_col: Optional[str] = args.label_col

    print("=== GNN PyG PREPROCESSING ===")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Processed dir: {PROCESSED_DIR}")
    print(f"SMILES col   : {smiles_col}")
    print(f"Label col    : {label_col!r}")

    # TRAIN_IN and VAL_OOD and TEST_OOD all come from chemistry pipeline splits
    print(f"[read] TRAIN_IN from {BLOCKS_TR_OUT}")
    train_graphs, train_stats = build_graphs_from_parquet(
        parquet_path=BLOCKS_TR_OUT,
        smiles_col=smiles_col,
        label_col=label_col,
    )

    print(f"[read] VAL_OOD from {BLOCKS_VA_OUT}")
    val_graphs, val_stats = build_graphs_from_parquet(
        parquet_path=BLOCKS_VA_OUT,
        smiles_col=smiles_col,
        label_col=label_col,
    )

    print(f"[read] TEST_OOD from {BLOCKS_TE_OUT}")
    test_graphs, test_stats = build_graphs_from_parquet(
        parquet_path=BLOCKS_TE_OUT,
        smiles_col=smiles_col,
        label_col=label_col,
    )

    print("\n[stats] TRAIN_IN :", train_stats)
    print("[stats] VAL_OOD  :", val_stats)
    print("[stats] TEST_OOD :", test_stats)

    # Quick sanity on feature dims
    if len(train_graphs) == 0:
        raise RuntimeError("No graphs were built for TRAIN_IN; check your data and columns.")

    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr.numel() > 0 else EDGE_FEAT_DIM

    print(f"\n[node/edge dims] NODE_FEAT_DIM={NODE_FEAT_DIM} (data: {node_dim}) "
          f"EDGE_FEAT_DIM={EDGE_FEAT_DIM} (data: {edge_dim})")

    # Save graph lists
    print(f"\n[write] {GNN_TRAIN_OUT}")
    torch.save(train_graphs, GNN_TRAIN_OUT)

    print(f"[write] {GNN_VAL_OUT}")
    torch.save(val_graphs, GNN_VAL_OUT)

    print(f"[write] {GNN_TEST_OUT}")
    torch.save(test_graphs, GNN_TEST_OUT)

    # Save metadata
    meta: Dict[str, Any] = {
        "node_feat_dim": int(node_dim),
        "edge_feat_dim": int(edge_dim),
        "smiles_col": smiles_col,
        "label_col": label_col,
        "stats": {
            "train_in": train_stats,
            "val_ood": val_stats,
            "test_ood": test_stats,
        },
        "source_parquets": {
            "train_in": str(BLOCKS_TR_OUT.resolve()),
            "val_ood": str(BLOCKS_VA_OUT.resolve()),
            "test_ood": str(BLOCKS_TE_OUT.resolve()),
        },
    }
    joblib.dump(meta, META_OUT)
    print(f"[write] metadata -> {META_OUT}")
    print("=== DONE GNN PyG PREPROCESSING ===")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess SMILES into PyG graphs for GNNs (BELKA BRD4)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="Build PyG graph lists for train/val/test splits")
    pp.add_argument("--smiles-col", type=str, default="smiles_clean",
                    help="Name of SMILES column in parquet files (default: 'smiles_clean')")
    pp.add_argument("--label-col", type=str, default="binds",
                    help="Name of binary label column in parquet files (default: 'binds')")
    pp.set_defaults(func=run_prepare)

    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
