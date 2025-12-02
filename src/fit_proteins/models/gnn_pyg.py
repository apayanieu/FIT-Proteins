#!/usr/bin/env python
"""
PyTorch Geometric GNN for BRD4 binding (BELKA 50k).

Assumes you have already run:

  python scripts/gnn_pyg_preproc.py prepare --label-col binds

which creates:

  data/processed/gnn_pyg_train_in.pt
  data/processed/gnn_pyg_val_ood.pt
  data/processed/gnn_pyg_test_ood.pt
  data/processed/gnn_pyg_metadata.joblib

This script:

  - loads those graph lists
  - builds a GNN with virtual node + GINEConv + sum pooling
  - trains on train_in, selects best epoch on val_ood (by AP)
  - reports AP / ROC-AUC on test_ood
  - saves best model + a text summary

CLI example:

  python src/fit_proteins/models/gnn_pyg.py train_gnn \
      --data-dir data/processed \
      --results-dir results/gnn_results \
      --epochs 35 \
      --batch-size 256
"""

import argparse
import math
import random
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool


# --------------------------------------------------------------------------------------
# Paths & helpers
# --------------------------------------------------------------------------------------

def get_project_root() -> Path:
    """Compute project root whether run as script or interactively."""
    try:
        # src/fit_proteins/models -> parents[2] -> project root
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


PROJECT_ROOT = get_project_root()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility (as much as PyTorch allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------------------
# Dataset wrapper
# --------------------------------------------------------------------------------------

class GraphListDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: List[Data]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


# --------------------------------------------------------------------------------------
# GNN model with virtual node
# --------------------------------------------------------------------------------------

class GNNWithVirtualNode(nn.Module):
    """
    GNN with:
      - linear encoders for node & edge features
      - stack of GINEConv layers
      - virtual node updated after each layer (except last)
      - sum global pooling
      - MLP head -> single logit
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(mlp)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # virtual node embedding
        self.virtualnode_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0.0)

        # MLPs to update virtual node after each layer (except last)
        self.mlp_virtual_list = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.mlp_virtual_list.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        # final prediction head on sum pooling
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Encode raw features
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr) if edge_attr is not None and edge_attr.numel() > 0 \
            else None

        # Initialize virtual node embedding per graph
        num_graphs = int(batch.max().item()) + 1
        virtualnode_emb = self.virtualnode_embedding.weight[0].unsqueeze(0).repeat(
            num_graphs, 1
        )  # [num_graphs, hidden_dim]

        for layer in range(self.num_layers):
            h = h + virtualnode_emb[batch]  # inject virtual node info

            if e is not None:
                h = self.convs[layer](h, edge_index, e)
            else:
                h = self.convs[layer](h, edge_index)

            h = self.bns[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # update virtual node embedding (except last layer)
            if layer < self.num_layers - 1:
                pooled = global_add_pool(h, batch)  # [num_graphs, hidden_dim]
                virtualnode_emb = virtualnode_emb + self.mlp_virtual_list[layer](pooled)

        # global sum pooling
        hg = global_add_pool(h, batch)

        logits = self.mlp_head(hg).squeeze(1)  # [num_graphs]
        return logits


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------

def load_graph_splits(data_dir: Path):
    """Load train/val/test graph lists + metadata."""
    meta_path = data_dir / "gnn_pyg_metadata.joblib"
    train_path = data_dir / "gnn_pyg_train_in.pt"
    val_path = data_dir / "gnn_pyg_val_ood.pt"
    test_path = data_dir / "gnn_pyg_test_ood.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("One or more *.pt graph files are missing in data_dir.")

    meta = joblib.load(meta_path)
    train_graphs: List[Data] = torch.load(train_path)
    val_graphs: List[Data] = torch.load(val_path)
    test_graphs: List[Data] = torch.load(test_path)

    return train_graphs, val_graphs, test_graphs, meta


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate AP and ROC-AUC on a given dataloader.
    Returns (AP, ROC_AUC). ROC_AUC may be NaN if only one class is present.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y.view(-1).cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    if np.all(y_true == 0) or np.all(y_true == 1):
        roc = float("nan")
    else:
        roc = roc_auc_score(y_true, y_prob)

    ap = average_precision_score(y_true, y_prob)
    return float(ap), float(roc)


def run_train_gnn(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir) if args.data_dir is not None else (PROJECT_ROOT / "data" / "processed")

    # default: put GNN results in results/gnn_results
    results_dir = (
        Path(args.results_dir)
        if args.results_dir is not None
        else (PROJECT_ROOT / "results" / "gnn_results")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"=== GNN-PyG TRAINING ===")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {data_dir}")
    print(f"Results dir  : {results_dir}")
    print(f"Device       : {device}")
    print(f"Seed         : {args.seed}")

    # --- Load graphs ---
    train_graphs, val_graphs, test_graphs, meta = load_graph_splits(data_dir)

    node_in_dim = meta["node_feat_dim"]
    edge_in_dim = meta["edge_feat_dim"]

    train_stats = meta["stats"]["train_in"]
    val_stats = meta["stats"]["val_ood"]
    test_stats = meta["stats"]["test_ood"]

    print(
        f"[stats] train_pos_rate={train_stats.get('pos_rate', float('nan')):.6f}  "
        f"val_pos_rate={val_stats.get('pos_rate', float('nan')):.6f}  "
        f"test_pos_rate={test_stats.get('pos_rate', float('nan')):.6f}"
    )

    print(
        f"[dims] node_in_dim={node_in_dim}  "
        f"edge_in_dim={edge_in_dim}"
    )

    # --- BCEWithLogits with class weighting ---
    n_pos = train_stats.get("n_pos", None)
    n_neg = train_stats.get("n_neg", None)
    if n_pos is not None and n_neg is not None and n_pos > 0:
        pos_weight_value = float(n_neg) / float(n_pos)
    else:
        pos_weight_value = 1.0

    print(f"[loss] BCEWithLogits pos_weight={pos_weight_value:.3f}")

    pos_weight = torch.tensor([pos_weight_value], device=device)

    # --- DataLoaders ---
    train_loader = DataLoader(
        GraphListDataset(train_graphs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        GraphListDataset(val_graphs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        GraphListDataset(test_graphs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Model, loss, optimizer ---
    model = GNNWithVirtualNode(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_ap = -1.0
    best_val_roc_at_best_ap = float("nan")
    best_epoch = 0
    best_model_path = results_dir / "gnn_pyg_best.pt"

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_graphs = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            targets = batch.y.view(-1)

            loss = criterion(logits, targets.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

        avg_loss = running_loss / max(1, n_graphs)

        # Evaluate on val_ood
        val_ap, val_roc = evaluate(model, val_loader, device)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={avg_loss:.4f}  "
            f"val_AP={val_ap:.4f}  "
            f"val_ROC={val_roc:.4f}"
        )

        # Track best model by val AP
        if not math.isnan(val_ap) and val_ap > best_val_ap:
            best_val_ap = val_ap
            best_val_roc_at_best_ap = val_roc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": meta,
                    "epoch": epoch,
                    "val_ap": val_ap,
                    "val_roc": val_roc,
                    "node_in_dim": node_in_dim,
                    "edge_in_dim": edge_in_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
                best_model_path,
            )
            print(f"  -> new best model saved to {best_model_path}")

    print(
        f"Training finished. Best val_AP={best_val_ap:.4f} "
        f"(val_ROC={best_val_roc_at_best_ap:.4f}) at epoch {best_epoch}"
    )

    # --- Final test evaluation using best model ---
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_ap, test_roc = evaluate(model, test_loader, device)
    print(f"[TEST_OOD] AP={test_ap:.4f}  ROC_AUC={test_roc:.4f}")

    # Save a tiny text log
    log_path = results_dir / "summary.txt"
    with open(log_path, "w") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best val_AP: {best_val_ap:.6f}\n")
        f.write(f"Best val_ROC_AUC_at_best_AP: {best_val_roc_at_best_ap:.6f}\n")
        f.write(f"Test_AP: {test_ap:.6f}\n")
        f.write(f"Test_ROC_AUC: {test_roc:.6f}\n")
        f.write(f"Train_pos_rate: {train_stats.get('pos_rate', float('nan')):.6f}\n")
        f.write(f"Val_pos_rate: {val_stats.get('pos_rate', float('nan')):.6f}\n")
        f.write(f"Test_pos_rate: {test_stats.get('pos_rate', float('nan')):.6f}\n")
        f.write(f"pos_weight: {pos_weight_value:.6f}\n")
        f.write(f"hidden_dim: {args.hidden_dim}\n")
        f.write(f"num_layers: {args.num_layers}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"epochs: {args.epochs}\n")

    print(f"[write] summary -> {log_path}")
    print("=== DONE GNN-PyG TRAINING ===")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PyG GNN for BRD4 binding (BELKA 50k)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train_gnn", help="Train & evaluate GNN on preprocessed graph data")
    pt.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with gnn_pyg_*.pt and metadata (default: data/processed)",
    )
    pt.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to store model & logs (default: results/gnn_results)",
    )
    pt.add_argument("--epochs", type=int, default=35)
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight-decay", type=float, default=1e-4)
    pt.add_argument("--hidden-dim", type=int, default=128)
    pt.add_argument("--num-layers", type=int, default=4)
    pt.add_argument("--dropout", type=float, default=0.2)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--num-workers", type=int, default=0)
    pt.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )

    pt.set_defaults(func=run_train_gnn)
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
