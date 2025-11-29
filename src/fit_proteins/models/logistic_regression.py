# src/fit_proteins/models/logistic_regression.py
# PyTorch logistic regression that reads your processed *.npz/*.npy files.
# Works with sparse CSR X matrices saved as .npz and y as 0/1 in .npy.

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_X(path: Path) -> sp.csr_matrix:
    # Expect a SciPy .npz saved via scipy.sparse.save_npz
    X = sp.load_npz(path)
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    return X


def load_y(path: Path) -> np.ndarray:
    y = np.load(path)
    # Ensure {0,1} float
    y = y.astype(np.float32)
    return y


class CSRDataset(Dataset):
    """Minimal dataset that returns dense rows on the fly to keep memory small."""
    def __init__(self, X: sp.csr_matrix, y: np.ndarray | None = None):
        self.X = X.tocsr()
        self.y = None if y is None else y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        x = torch.from_numpy(row.toarray().ravel().astype(np.float32))  # [n_features]
        if self.y is None:
            return x
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


class LogReg(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        logits = self.linear(x).squeeze(1)  # [batch]
        return logits  # use with BCEWithLogitsLoss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        xb, yb = batch
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy()
        y_prob.append(prob)
        y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred))
    }
    # AUC may fail if only one class present — guard it.
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report_macro_f1"] = float(report["macro avg"]["f1-score"])
    return metrics


def main(
    x_train_path: str,
    y_train_path: str,
    x_test_path: str | None = None,
    ids_test_path: str | None = None,
    out_dir: str = "artifacts",
    batch_size: int = 4096,
    epochs: int = 10,
    lr: float = 1e-2,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_path = Path(x_train_path)
    y_train_path = Path(y_train_path)
    x_test_path = Path(x_test_path) if x_test_path else None
    ids_test_path = Path(ids_test_path) if ids_test_path else None
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
    X_full = load_X(x_train_path)
    y_full = load_y(y_train_path)
    n_features = X_full.shape[1]

    # Split train/val
    stratify = y_full if len(np.unique(y_full)) > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=seed, stratify=stratify
    )

    train_ds = CSRDataset(X_tr, y_tr)
    val_ds   = CSRDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ---------- Model ----------
    model = LogReg(n_features).to(device)

    # Optional class-imbalance handling with pos_weight
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    pos_weight = None
    if pos > 0 and neg > 0:
        pos_weight = torch.tensor([neg / pos], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------- Train ----------
    best = {"val_loss": float("inf")}
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            bsz = yb.size(0)
            running += loss.item() * bsz
            seen += bsz
        train_loss = running / max(seen, 1)

        # Eval
        model.eval()
        v_running, v_seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                bsz = yb.size(0)
                v_running += loss.item() * bsz
                v_seen += bsz
        val_loss = v_running / max(v_seen, 1)
        metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={metrics.get('accuracy', float('nan')):.4f}  "
              f"auc={metrics.get('roc_auc', float('nan')):.4f}")

        # Save best
        if val_loss < best["val_loss"]:
            best.update({"val_loss": val_loss})
            torch.save(model.state_dict(), out_dir / "pytorch_logreg.pt")

    # Final metrics
    final_metrics = evaluate(model, val_loader, device)
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
    print("\nSaved:", (out_dir / "pytorch_logreg.pt").resolve())
    print("Metrics:", final_metrics)

    # ---------- Optional test inference ----------
    if x_test_path:
        X_test = load_X(x_test_path)
        test_ds = CSRDataset(X_test, None)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        model.eval()
        probs = []
        with torch.no_grad():
            for xb in test_loader:
                xb = xb.to(device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
                probs.append(p)
        probs = np.concatenate(probs).ravel()
        np.save(out_dir / "test_proba.npy", probs)
        print("Saved:", (out_dir / "test_proba.npy").resolve())

        # If IDs provided, write a CSV pairing ids with probabilities
        if ids_test_path and ids_test_path.exists():
            import pandas as pd
            ids = np.load(ids_test_path)
            df = pd.DataFrame({"id": ids, "proba": probs})
            df.to_csv(out_dir / "test_predictions.csv", index=False)
            print("Saved:", (out_dir / "test_predictions.csv").resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PyTorch Logistic Regression (sparse CSR-friendly)")
    p.add_argument("--x-train", default="processed/X_train_full.npz")
    p.add_argument("--y-train", default="processed/y_train_full.npy")
    p.add_argument("--x-test", default="processed/X_test.npz")
    p.add_argument("--ids-test", default="processed/ids_test.npy")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(
        x_train_path=args.x_train,
        y_train_path=args.y_train,
        x_test_path=args.x_test,
        ids_test_path=args.ids_test,
        out_dir=args.out,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
