#!/usr/bin/env python
"""
XGBoost on ECFP features for BRD4 binding prediction.

This script runs a small XGBoost experiment sweep over multiple hyperparameter
combinations and random seeds, using the block-aware BRD4 split:

    • train_in – molecules whose building blocks appear in the training chemistry domain
    • val_ood – molecules containing at least one building block NOT seen during training
    • test_ood – molecules reserved for final OOD evaluation

For each XGBoost configuration and random seed, it:

1. Trains on train_in with (attempted) early stopping on val_ood:
       – Average Precision (AP): primary metric
       – ROC-AUC: secondary metric
2. Records the best_iteration (if early stopping is available).
   If early stopping is NOT supported by this xgboost version, it falls back
   to using the full n_estimators.
3. Retrains on the combined set (train_in + val_ood) using
   n_estimators = best_iteration + 1 (or the full count if no early stopping).
4. Evaluates the final model on test_ood (AP and AUC).

All runs are collected into a results table and stored in a CSV file.

If the results file already exists, new runs are APPENDED to it.
If it does not exist, it is created.

Usage (from project root):
    python src/fit_proteins/models/xgb_ecfp.py xgb \
        --data-dir data/processed \
        --results-dir results/xgb_ecfp
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack

import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path!s}")


def load_split(data_dir: Path):
    """Load pre-split ECFP matrices and labels for train_in, val_ood, test_ood."""
    X_train_in_fp = data_dir / "X_train_in_ecfp.npz"
    y_train_in_fp = data_dir / "y_train_in.npy"

    X_val_ood_fp = data_dir / "X_val_ood_ecfp.npz"
    y_val_ood_fp = data_dir / "y_val_ood.npy"

    X_test_ood_fp = data_dir / "X_test_ood_ecfp.npz"
    y_test_ood_fp = data_dir / "y_test_ood.npy"

    for p in (
        X_train_in_fp,
        y_train_in_fp,
        X_val_ood_fp,
        y_val_ood_fp,
        X_test_ood_fp,
        y_test_ood_fp,
    ):
        ensure_exists(p)

    X_tr = load_npz(X_train_in_fp)
    y_tr = np.load(y_train_in_fp)

    X_val = load_npz(X_val_ood_fp)
    y_val = np.load(y_val_ood_fp)

    X_test = load_npz(X_test_ood_fp)
    y_test = np.load(y_test_ood_fp)

    return X_tr, y_tr, X_val, y_val, X_test, y_test


def binary_stats(y: np.ndarray, name: str) -> None:
    """Print basic class balance statistics."""
    n = y.shape[0]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_rate = n_pos / n if n > 0 else float("nan")
    print(f"[{name}] n={n} | positives={n_pos} | negatives={n_neg} | pos_rate={pos_rate:.6f}")


def compute_ap_auc(y_true: np.ndarray, y_score: np.ndarray, prefix: str):
    """Compute and print AP and ROC-AUC."""
    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    print(f" {prefix} AP:  {ap:.6f}")
    print(f" {prefix} AUC: {auc:.6f}")
    return ap, auc


# -------------------------------------------------------------------
# Core XGBoost routine
# -------------------------------------------------------------------
def run_xgb(args):
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== LOADING DATA ===")
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_split(data_dir)

    print("Train_in:", X_tr.shape, "| Pos:", int(y_tr.sum()))
    print("Val_ood :", X_val.shape, "| Pos:", int(y_val.sum()))
    print("Test_ood:", X_test.shape, "| Pos:", int(y_test.sum()))
    print(
        "Positive rates → "
        f"train_in: {y_tr.mean():.6f} | "
        f"val_ood: {y_val.mean():.6f} | "
        f"test_ood: {y_test.mean():.6f}"
    )

    binary_stats(y_tr, "train_in")
    binary_stats(y_val, "val_ood")
    binary_stats(y_test, "test_ood")

    # ----------------------------------------------------------------
    # Class imbalance weight (from train_in only)
    # ----------------------------------------------------------------
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"\nscale_pos_weight (from train_in): {scale_pos_weight:.4f}")

    # ----------------------------------------------------------------
    # XGBoost configurations and seeds
    # ----------------------------------------------------------------
    # Base settings shared by all configs
    xgb_base = dict(
        objective="binary:logistic",
        tree_method="hist",          # if this ever errors, set to "auto"
        n_estimators=2000,           # upper bound; early stopping may cut this
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",         # set here, not in fit()
        n_jobs=-1,
        random_state=0,
    )

    # Four coherent configurations around a tuned regime
    xgb_configs = [
        dict(
            name="xgb_lr0.05_depth7_cols0.6",
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
        ),
        dict(
            name="xgb_lr0.05_depth9_cols0.6",
            learning_rate=0.05,
            max_depth=9,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
        ),
        dict(
            name="xgb_lr0.03_depth8_cols0.8",
            learning_rate=0.03,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
        ),
        dict(
            name="xgb_lr0.10_depth6_cols0.6_l2",
            learning_rate=0.10,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=1.0,
            reg_lambda=2.0,
            reg_alpha=0.0,
        ),
    ]

    seeds = [0, 1, 2]

    print("\n=== RUNNING XGBOOST EXPERIMENTS (4 configs × 3 seeds) ===")
    rows = []

    for cfg_idx, cfg in enumerate(xgb_configs):
        print(f"\n--- Config {cfg_idx + 1}/{len(xgb_configs)}: {cfg['name']} ---")

        for seed in seeds:
            print(f" Seed {seed}:")

            # Full parameter set for this run
            params = dict(xgb_base)
            params.update(
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                min_child_weight=cfg["min_child_weight"],
                reg_lambda=cfg["reg_lambda"],
                reg_alpha=cfg["reg_alpha"],
                random_state=seed,
            )

            np.random.seed(seed)

            # ---------------------------------------------------
            # Step 1: Train on train_in, TRY early stopping on val_ood
            # ---------------------------------------------------
            model = xgb.XGBClassifier(**params)

            try:
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_tr, y_tr), (X_val, y_val)],
                    early_stopping_rounds=100,
                    verbose=False,
                )
                best_iter = getattr(model, "best_iteration", None)
                if best_iter is None:
                    best_iter = getattr(
                        model, "best_ntree_limit", model.n_estimators
                    ) - 1
                used_es = True
                print("  early_stopping_rounds supported; using best_iteration.")
            except TypeError:
                # Older xgboost: no early_stopping_rounds in fit()
                print("  early_stopping_rounds NOT supported by this xgboost version.")
                print("  Training with fixed n_estimators (no early stopping).")
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr)
                best_iter = params["n_estimators"] - 1
                used_es = False

            n_estimators_final = int(best_iter) + 1
            print(
                f"  best_iteration={best_iter} → "
                f"n_estimators_final={n_estimators_final} "
                f"(used_early_stopping={used_es})"
            )

            # Metrics on train_in and val_ood using the model
            train_proba = model.predict_proba(X_tr)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]

            ap_train, auc_train = compute_ap_auc(y_tr, train_proba, "Train_in")
            ap_val, auc_val = compute_ap_auc(y_val, val_proba, "Val_ood")

            # ---------------------------------------------------
            # Step 2: Retrain on train_in + val_ood with n_estimators_final
            # ---------------------------------------------------
            X_all = vstack([X_tr, X_val])
            y_all = np.concatenate([y_tr, y_val])

            final_params = dict(params)
            final_params["n_estimators"] = n_estimators_final

            final_model = xgb.XGBClassifier(**final_params)
            final_model.fit(X_all, y_all)

            # ---------------------------------------------------
            # Step 3: Evaluate on test_ood
            # ---------------------------------------------------
            test_proba = final_model.predict_proba(X_test)[:, 1]
            ap_test, auc_test = compute_ap_auc(y_test, test_proba, "Test_ood")

            rows.append(
                dict(
                    config_name=cfg["name"],
                    seed=seed,
                    n_estimators_max=xgb_base["n_estimators"],
                    n_estimators_final=n_estimators_final,
                    used_early_stopping=used_es,
                    learning_rate=cfg["learning_rate"],
                    max_depth=cfg["max_depth"],
                    subsample=cfg["subsample"],
                    colsample_bytree=cfg["colsample_bytree"],
                    min_child_weight=cfg["min_child_weight"],
                    reg_lambda=cfg["reg_lambda"],
                    reg_alpha=cfg["reg_alpha"],
                    scale_pos_weight=scale_pos_weight,
                    ap_train=ap_train,
                    auc_train=auc_train,
                    ap_val=ap_val,
                    auc_val=auc_val,
                    ap_test=ap_test,
                    auc_test=auc_test,
                )
            )

    # ----------------------------------------------------------------
    # Build and append to results table
    # ----------------------------------------------------------------
    df_new = pd.DataFrame(rows)

    print("\n=== NEW RESULTS TABLE (this run) ===")
    print(df_new.to_string(index=False))

    out_path = results_dir / "xgb_ecfp_experiments.csv"
    if out_path.exists():
        print(f"\nExisting results file found at: {out_path}, appending new runs.")
        df_old = pd.read_csv(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        print(f"\nNo existing results file found. Creating new one at: {out_path}")
        df_all = df_new

    df_all.to_csv(out_path, index=False)
    print(f"Saved combined results table to: {out_path}")
    print("\nDone.")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="XGBoost on ECFP features for BRD4 (OOD split)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    xgb_parser = sub.add_parser("xgb", help="Run XGBoost experiment sweep on ECFP.")
    xgb_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with pre-split ECFP arrays (relative to project root).",
    )
    xgb_parser.add_argument(
        "--results-dir",
        type=str,
        default="results/xgb_ecfp",
        help="Directory to store the results table (CSV).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "xgb":
        run_xgb(args)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
