#!/usr/bin/env python
"""
SMILES-CNN for BRD4 binding (BELKA 50k).

Reads preprocessed NN data produced by scripts/smiles_nn_preproc.py:

  data/processed/smiles_cnn_train_in.npz
  data/processed/smiles_cnn_val_ood.npz
  data/processed/smiles_cnn_test_ood.npz
  data/processed/smiles_cnn_metadata.joblib

Trains a 1D CNN over tokenized SMILES with focal loss (fixed alpha),
reports Average Precision (AP) on val_ood and test_ood, and saves the best model.

CLI example:

  python src/fit_proteins/models/smiles_cnn.py train_cnn \
      --data-dir data/processed \
      --results-dir results/cnn_results \
      --epochs 20 \
      --batch-size 256
"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


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
# Dataset & model
# --------------------------------------------------------------------------------------

class SmilesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: [N, L] integer token IDs
        y: [N] binary labels (0/1)
        """
        self.X = torch.as_tensor(X, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SmilesCNN(nn.Module):
    """
    Simple SMILES-CNN:
      - Embedding
      - Multiple Conv1d filters with different kernel sizes
      - Global max pooling
      - MLP head -> single logit
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embed_dim: int = 128,
        num_filters: int = 256,
        kernel_sizes=(3, 5, 7),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] integer tokens
        returns logits: [B]
        """
        # [B, L, E]
        emb = self.embedding(x)
        # [B, E, L]
        emb = emb.transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            # [B, C, L'] after conv
            c = F.relu(conv(emb))
            # Global max pool over sequence -> [B, C]
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            conv_outs.append(p)

        h = torch.cat(conv_outs, dim=1)
        h = self.dropout(h)
        logits = self.fc(h).squeeze(1)  # [B]
        return logits


class FocalLossFixedAlpha(nn.Module):
    """
    Focal loss with fixed α for the positive class.

    For binary classification:

      L = - α_t (1 - p_t)^γ log(p_t)

      where:
        p = sigmoid(logit)
        p_t = p      if y=1
              1 - p  if y=0

      α_t = alpha_pos    if y=1
            alpha_neg    if y=0
      with alpha_neg = 1 - alpha_pos
    """

    def __init__(self, alpha_pos: float = 0.9, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        if not (0.0 < alpha_pos < 1.0):
            raise ValueError("alpha_pos must be in (0, 1).")
        self.alpha_pos = float(alpha_pos)
        self.alpha_neg = 1.0 - float(alpha_pos)
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B] raw scores
        targets: [B] in {0,1}
        """
        targets = targets.float()
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        # α_t per sample
        alpha = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.alpha_pos),
            torch.full_like(targets, self.alpha_neg),
        )

        # p_t
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)

        loss = -alpha * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        return loss.mean()


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------

def load_data(data_dir: Path) -> Tuple[SmilesDataset, SmilesDataset, SmilesDataset, dict]:
    """Load train/val/test npz + metadata."""
    meta_path = data_dir / "smiles_cnn_metadata.joblib"
    train_path = data_dir / "smiles_cnn_train_in.npz"
    val_path = data_dir / "smiles_cnn_val_ood.npz"
    test_path = data_dir / "smiles_cnn_test_ood.npz"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("One or more .npz files are missing in data_dir.")

    meta = joblib.load(meta_path)

    tr = np.load(train_path)
    va = np.load(val_path)
    te = np.load(test_path)

    X_tr, y_tr = tr["X"], tr["y"]
    X_va, y_va = va["X"], va["y"]
    X_te, y_te = te["X"], te["y"]

    ds_tr = SmilesDataset(X_tr, y_tr)
    ds_va = SmilesDataset(X_va, y_va)
    ds_te = SmilesDataset(X_te, y_te)

    return ds_tr, ds_va, ds_te, meta


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
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    if np.all(y_true == 0) or np.all(y_true == 1):
        # ROC is not defined when there's only one class
        roc = float("nan")
    else:
        roc = roc_auc_score(y_true, y_prob)

    ap = average_precision_score(y_true, y_prob)
    return float(ap), float(roc)


def run_train_cnn(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir) if args.data_dir is not None else (PROJECT_ROOT / "data" / "processed")

    # >>> changed default here: put CNN results in results/cnn_results <<<
    results_dir = (
        Path(args.results_dir)
        if args.results_dir is not None
        else (PROJECT_ROOT / "results" / "cnn_results")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"=== SMILES-CNN TRAINING ===")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {data_dir}")
    print(f"Results dir  : {results_dir}")
    print(f"Device       : {device}")
    print(f"Seed         : {args.seed}")

    # --- Load data ---
    ds_tr, ds_va, ds_te, meta = load_data(data_dir)

    vocab_size = len(meta["idx2token"])
    pad_idx = meta["pad_idx"]
    train_pos_rate = meta["stats"]["train_pos_rate"]
    val_pos_rate = meta["stats"]["val_pos_rate"]
    test_pos_rate = meta["stats"]["test_pos_rate"]

    print(
        f"[stats] train_pos_rate={train_pos_rate:.6f}  "
        f"val_pos_rate={val_pos_rate:.6f}  "
        f"test_pos_rate={test_pos_rate:.6f}"
    )

    print(
        f"[loss] focal alpha_pos={args.alpha_pos:.3f}  "
        f"alpha_neg={1.0 - args.alpha_pos:.3f}  "
        f"gamma={args.gamma:.2f}"
    )

    train_loader = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model, loss, optimizer ---
    model = SmilesCNN(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        kernel_sizes=tuple(args.kernel_sizes),
        dropout=args.dropout,
    ).to(device)

    criterion = FocalLossFixedAlpha(
        alpha_pos=args.alpha_pos,
        gamma=args.gamma,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_ap = 0.0
    best_epoch = 0
    best_model_path = results_dir / "smiles_cnn_best.pt"

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)

        # Evaluate on val_ood
        val_ap, val_roc = evaluate(model, val_loader, device)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={avg_loss:.4f}  "
            f"val_AP={val_ap:.4f}  "
            f"val_ROC={val_roc:.4f}"
        )

        # Track best model by val AP
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": meta,
                    "epoch": epoch,
                    "val_ap": val_ap,
                    "val_roc": val_roc,
                },
                best_model_path,
            )
            print(f"  -> new best model saved to {best_model_path}")

    print(f"Training finished. Best val_AP={best_val_ap:.4f} at epoch {best_epoch}")

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
        f.write(f"Test_AP: {test_ap:.6f}\n")
        f.write(f"Test_ROC_AUC: {test_roc:.6f}\n")
        f.write(f"Train_pos_rate: {train_pos_rate:.6f}\n")
        f.write(f"Val_pos_rate: {val_pos_rate:.6f}\n")
        f.write(f"Test_pos_rate: {test_pos_rate:.6f}\n")
        f.write(f"alpha_pos: {args.alpha_pos:.6f}\n")
        f.write(f"alpha_neg: {1.0 - args.alpha_pos:.6f}\n")
        f.write(f"gamma: {args.gamma:.6f}\n")

    print(f"[write] summary -> {log_path}")
    print("=== DONE SMILES-CNN TRAINING ===")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SMILES-CNN for BRD4 binding (BELKA 50k)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train_cnn", help="Train & evaluate SMILES-CNN on preprocessed data")
    pt.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with smiles_cnn_*.npz and metadata (default: data/processed)",
    )
    pt.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to store model & logs (default: results/cnn_results)",
    )
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight-decay", type=float, default=0.0)
    pt.add_argument("--embed-dim", type=int, default=128)
    pt.add_argument("--num-filters", type=int, default=256)
    pt.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="Kernel sizes for Conv1d filters",
    )
    pt.add_argument("--dropout", type=float, default=0.2)
    pt.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (default 2.0)",
    )
    pt.add_argument(
        "--alpha-pos",
        type=float,
        default=0.9,
        help="Focal loss α for positive class; negative gets 1-α (default 0.9)",
    )
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--num-workers", type=int, default=0)
    pt.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )

    pt.set_defaults(func=run_train_cnn)
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
