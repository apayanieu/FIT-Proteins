#!/usr/bin/env python
"""
Random Forest on ECFP features for BRD4 binding prediction.

This script runs a small RF experiment sweep over multiple hyperparameter
combinations and random seeds, using the block-aware BRD4 split:

    • train_in  – molecules whose building blocks appear in the training chemistry domain
    • val_ood   – molecules containing at least one building block NOT seen during training
    • test_ood  – molecules reserved for final OOD evaluation

For each RF configuration and random seed, it:

1. Trains on train_in and evaluates on val_ood:
       – Average Precision (AP): primary metric
       – ROC-AUC: secondary metric
       – OOB score on train_in (sanity check; in-domain, not OOD)
2. Retrains on the combined set (train_in + val_ood).
3. Evaluates the final model on test_ood:
       – AP and AUC

All runs are collected into a results table and saved as a CSV file.

Usage:
    python rf_ecfp.py rf --data-dir data/processed --results-dir results/rf_ecfp

If --results-dir does not exist, it will be created.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack

from sklearn.ensemble import RandomForestClassifier
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


# -------------------------------------------------------------------
# Core RF routine
# -------------------------------------------------------------------
def run_rf(args):
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

    # ----------------------------------------------------------------
    # RF configurations and seeds
    # ----------------------------------------------------------------
    # Base RF settings shared by all configs
    rf_base = dict(
        class_weight="balanced_subsample",
        criterion="log_loss",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
    )

    # Four RF configurations to probe around the tuned regime
    rf_configs = [
        dict(
            name="rf_400_sqrt_leaf1",
            n_estimators=400,
            max_features="sqrt",
            max_depth=None,
            min_samples_leaf=1,
        ),
        dict(
            name="rf_800_sqrt_leaf1",  # close to the tuned config you found
            n_estimators=800,
            max_features="sqrt",
            max_depth=None,
            min_samples_leaf=1,
        ),
        dict(
            name="rf_800_sqrt_leaf3",
            n_estimators=800,
            max_features="sqrt",
            max_depth=32,
            min_samples_leaf=3,
        ),
        dict(
            name="rf_800_0.2_leaf5",
            n_estimators=800,
            max_features=0.2,
            max_depth=32,
            min_samples_leaf=5,
        ),
    ]

    seeds = [0, 1, 2]

    print("\n=== RUNNING RF EXPERIMENTS (4 configs × 3 seeds) ===")
    results = []

    for cfg_idx, cfg in enumerate(rf_configs):
        print(f"\n--- Config {cfg_idx + 1}/{len(rf_configs)}: {cfg['name']} ---")
        for seed in seeds:
            print(f"  Seed {seed}:")

            # Full parameter set for this run
            params = dict(rf_base)
            params.update(
                n_estimators=cfg["n_estimators"],
                max_features=cfg["max_features"],
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                random_state=seed,
            )

            # ---------------------------------------------------
            # Step 1: Train on train_in, evaluate on val_ood
            # ---------------------------------------------------
            rf = RandomForestClassifier(**params)

            # Set numpy seed for reproducibility outside RF
            np.random.seed(seed)

            rf.fit(X_tr, y_tr)

            oob_train_in = getattr(rf, "oob_score_", np.nan)

            val_proba = rf.predict_proba(X_val)[:, 1]
            ap_val = average_precision_score(y_val, val_proba)
            auc_val = roc_auc_score(y_val, val_proba)

            print(
                f"    train_in→val_ood | "
                f"OOB_train_in: {oob_train_in:.6f} | "
                f"AP_val: {ap_val:.6f} | AUC_val: {auc_val:.6f}"
            )

            # ---------------------------------------------------
            # Step 2: Retrain on train_in + val_ood
            # ---------------------------------------------------
            X_all = vstack([X_tr, X_val])
            y_all = np.concatenate([y_tr, y_val])

            rf_final = RandomForestClassifier(**params)
            rf_final.fit(X_all, y_all)

            oob_train_all = getattr(rf_final, "oob_score_", np.nan)

            # ---------------------------------------------------
            # Step 3: Evaluate on test_ood
            # ---------------------------------------------------
            test_proba = rf_final.predict_proba(X_test)[:, 1]
            ap_test = average_precision_score(y_test, test_proba)
            auc_test = roc_auc_score(y_test, test_proba)

            print(
                f"    train+val→test_ood | "
                f"OOB_train_all: {oob_train_all:.6f} | "
                f"AP_test: {ap_test:.6f} | AUC_test: {auc_test:.6f}"
            )

            results.append(
                dict(
                    config_name=cfg["name"],
                    seed=seed,
                    n_estimators=cfg["n_estimators"],
                    max_features=str(cfg["max_features"]),
                    max_depth="None" if cfg["max_depth"] is None else cfg["max_depth"],
                    min_samples_leaf=cfg["min_samples_leaf"],
                    oob_train_in=oob_train_in,
                    ap_val=ap_val,
                    auc_val=auc_val,
                    oob_train_all=oob_train_all,
                    ap_test=ap_test,
                    auc_test=auc_test,
                )
            )

    # ----------------------------------------------------------------
    # Build and save results table
    # ----------------------------------------------------------------
    df = pd.DataFrame(results)

    print("\n=== RESULTS TABLE (per config × seed) ===")
    # Full table
    print(df.to_string(index=False))

    out_path = results_dir / "rf_ecfp_experiments.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results table to: {out_path}")
    print("\nDone.")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Random Forest on ECFP features for BRD4 (OOD split)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    rf = sub.add_parser("rf", help="Run RF experiment sweep on ECFP.")
    rf.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with pre-split ECFP arrays (relative to project root).",
    )
    rf.add_argument(
        "--results-dir",
        type=str,
        default="results/rf_ecfp",
        help="Directory to store the results table (CSV).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "rf":
        run_rf(args)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
