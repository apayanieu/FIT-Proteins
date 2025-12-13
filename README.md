# F.I.T Proteins 🧬  
_Fast Inference of Therapeutic PROTEIN–Ligand Interactions for BRD4_

This repository contains the code and experiments for a BRD4 binding prediction project based on a 50k-molecule subset of the BELKA DNA-encoded library (DEL) dataset.

The core focus is **learning under extreme class imbalance** and **out-of-distribution (OOD) generalization** to molecules built from previously unseen combinatorial building blocks.

---

## Project overview

**Goal:**  
Given a molecule (as SMILES or a molecular graph), predict whether it binds to **BRD4**.

**Key challenges:**

- Very low active rate (~0.5% positives).
- Library artefacts (e.g. linker token) that can leak into models.
- Need to generalize to **new chemical blocks**, not just random splits.

**What this project implements:**

- Chemistry preprocessing:
  - Linker removal and SMILES canonicalization.
  - Extraction of combinatorial **building blocks**.
- Block-aware **OOD split**:
  - `train_in`: only training blocks.
  - `val_ood`: molecules with validation blocks.
  - `test_ood`: molecules with test-only blocks.
- Multiple feature views:
  - **ECFP** fingerprints (radius 2, 2048 bits).
  - **SMILES token sequences** for CNNs.
  - **Molecular graphs** (nodes + edges) for GNNs.
- Models:
  - Random Forest (RF) on ECFP.
  - XGBoost on ECFP.
  - SMILES-CNN with focal loss (fixed α).
  - GNN (PyTorch Geometric) with class-weighted BCE.
- OOD evaluation:
  - Metrics: **Average Precision (AP)** and **ROC-AUC** on both `val_ood` and `test_ood`.
  - Results stored under `results/` (CSV/TXT, heavy binaries ignored).

---

## Repository structure

```text
.
├── LICENSE
├── README.md              # You are here
├── environment.yml        # Conda environment (recommended)
├── pyproject.toml         # Package metadata & dependencies
├── data
│   ├── raw
│   │   └── train_brd4_50k_stratified.parquet
│   └── processed          # Precomputed splits & features
│       ├── train_brd4_50k_clean*.parquet
│       ├── X_*_ecfp.npz
│       ├── smiles_cnn_* .npz
│       ├── gnn_pyg_* .pt
│       ├── y_* .npy
│       └── ... metadata ...
├── notebooks
│   ├── data_exp.ipynb           # EDA / data inspection
│   ├── random_forest_presplit.ipynb
│   ├── xgboost_presplit.ipynb
│   ├── NNs.ipynb                # CNN experiments
│   ├── GNNs.ipynb               # GNN experiments
│   └── visuals.ipynb            # Figures for the report
├── results
│   ├── rf_ecfp
│   │   └── rf_ecfp_experiments.csv
│   ├── xgb_ecfp
│   │   └── xgb_ecfp_experiments.csv
│   ├── cnn_results
│   │   ├── smiles_cnn_best.pt   # ignored by git
│   │   └── summary.txt
│   └── gnn_results
│       ├── gnn_pyg_best.pt      # ignored by git
│       └── summary.txt
├── scripts
│   ├── chemistry_pipeline.py    # cleaning + block-aware splits
│   ├── data_stratified.py       # (subset/stratification helper)
│   ├── ecfp_gen.py              # ECFP feature generation
│   ├── gnn_pyg_gen.py           # graph dataset generation
│   └── smiles_nn_token.py       # SMILES tokenization for CNN
└── src
    └── fit_proteins
        ├── __about__.py
        ├── __init__.py
        ├── io
        ├── models
        │   ├── rf_efcp.py       # RF training & evaluation
        │   ├── xgb_ecfp.py      # XGB training & evaluation
        │   ├── smiles_cnn.py    # SMILES-CNN training & eval
        │   └── gnn_pyg.py       # GNN (PyG) training & eval
        └── utils
````

> **Note:** Model weights (`*.pt`, `*.npz`, `*.joblib`, etc.) are ignored by git; only the **results summaries** (CSV/TXT) are versioned.

---

## Installation

### 1. Conda (recommended)

This is the intended path, especially for RDKit, PyTorch, and PyG:

```bash
# from the project root
conda env create -f environment.yml
conda activate fit-proteins
```

This environment should contain everything needed to run the scripts and notebooks.

### 2. Editable install (optional)

If you want to install the package itself (e.g. for `python -m fit_proteins.models.*`):

```bash
pip install -e .[chem,gbm,dl,viz,io]
```

Adjust the extras according to what you actually need (e.g. if you only want RF/XGB, you don’t need `dl` or `chem`.

---

## Data

The repository ships with:

* `data/raw/train_brd4_50k_stratified.parquet`
  Stratified 50k subset of BELKA BRD4 library.

* `data/processed/`
  Contains cleaned Parquets, the block-based splits, and precomputed features & labels:

  * `train_brd4_50k_clean.parquet`, `train_brd4_50k_clean_blocks_*.parquet`
  * `X_train_in_ecfp.npz`, `X_val_ood_ecfp.npz`, `X_test_ood_ecfp.npz`
  * `smiles_cnn_* .npz`
  * `gnn_pyg_* .pt`
  * `y_* .npy`, `ids_* .npy`
  * Metadata `.joblib` files.

You can either **use these precomputed artefacts** directly, or regenerate them via the scripts below.

---

## Reproducing the pipeline

All commands assume:

```bash
conda activate fit-proteins
cd path/to/F.I.T-Proteins
```

### 1. Chemistry preprocessing & block-aware split

```bash
python scripts/chemistry_pipeline.py
```

What this does:

* Reads `data/raw/train_brd4_50k_stratified.parquet`.
* Removes the DEL linker token and canonicalizes SMILES.
* Extracts building blocks.
* Creates:

  * `train_brd4_50k_clean*.parquet`
  * `train_brd4_50k_clean_blocks_*.parquet`
  * Block-aware splits: `train_in`, `val_ood`, `test_ood`.

> `scripts/data_stratified.py` is used to build the 50k subset in the first place. You do not need it if you are starting from the provided `train_brd4_50k_stratified.parquet`.

### 2. Feature generation

#### 2.1 ECFP fingerprints

```bash
python scripts/ecfp_gen.py
```

Outputs (in `data/processed/`):

* `X_train_in_ecfp.npz`, `X_val_ood_ecfp.npz`, `X_test_ood_ecfp.npz`
* `y_train_in.npy`, `y_val_ood.npy`, `y_test_ood.npy`
* `ids_* .npy`, `ecfp_metadata.joblib`

#### 2.2 SMILES sequences for CNN

```bash
python scripts/smiles_nn_token.py
```

Outputs:

* `smiles_cnn_train_in.npz`
* `smiles_cnn_val_ood.npz`
* `smiles_cnn_test_ood.npz`
* `smiles_cnn_metadata.joblib`

#### 2.3 Graph datasets for GNN

```bash
python scripts/gnn_pyg_gen.py
```

Outputs:

* `gnn_pyg_train_in.pt`
* `gnn_pyg_val_ood.pt`
* `gnn_pyg_test_ood.pt`
* `gnn_pyg_metadata.joblib`

---

## Training and evaluation

All model scripts live under `src/fit_proteins/models`. You can run them with `python -m` from the project root (so that `src/` is on `PYTHONPATH`).

> The scripts are written as standalone modules with their own argument parsers and training loops. Check `--help` on each if you need more details.

### 1. Random Forest (ECFP)

```bash
python -m fit_proteins.models.rf_efcp
```

What it does:

* Loads ECFP features and labels from `data/processed/`.
* Trains multiple RF configurations on `train_in`.
* Evaluates AP and ROC-AUC on `val_ood` and `test_ood`.
* Writes results to:

  * `results/rf_ecfp/rf_ecfp_experiments.csv`

### 2. XGBoost (ECFP)

```bash
python -m fit_proteins.models.xgb_ecfp
```

What it does:

* Loads ECFP features and labels.
* Computes `scale_pos_weight` from `train_in`.
* Trains XGB models with early stopping on `val_ood` AP (where supported by the XGBoost version).
* Retrains on `train_in ∪ val_ood` and evaluates on `test_ood`.
* Writes:

  * `results/xgb_ecfp/xgb_ecfp_experiments.csv`

### 3. SMILES-CNN

```bash
python -m fit_proteins.models.smiles_cnn
```

What it does:

* Loads SMILES token datasets and metadata.
* Builds a Conv1d-based model with embedding, global pooling, and a linear output head.
* Uses **focal loss** with fixed α (for positive/negative classes) and focusing parameter γ.
* Trains on `train_in`, early stops on `val_ood` AP.
* Saves:

  * `results/cnn_results/smiles_cnn_best.pt` (weights, ignored by git)
  * `results/cnn_results/summary.txt` (metrics and settings)

### 4. Graph Neural Network (PyG)

```bash
python -m fit_proteins.models.gnn_pyg
```

What it does:

* Loads PyG datasets from `gnn_pyg_*.pt`.
* Builds a message-passing GNN with a **virtual node**, global pooling, and an MLP head.
* Uses **BCEWithLogitsLoss** with `pos_weight = N_neg / N_pos` for imbalance.
* Trains on `train_in`, early stops on `val_ood` AP.
* Saves:

  * `results/gnn_results/gnn_pyg_best.pt` (weights, ignored by git)
  * `results/gnn_results/summary.txt` (metrics and settings)

---

## Notebooks & visualization

The notebooks under `notebooks/` are for exploration and report figures, not the main pipeline:

* `data_exp.ipynb`: basic EDA on splits and labels.
* `random_forest_presplit.ipynb`, `xgboost_presplit.ipynb`: exploratory tree experiments.
* `NNs.ipynb`: SMILES-CNN experiments / debugging.
* `GNNs.ipynb`: GNN experiments / debugging.
* `visuals.ipynb`:
  Uses the **saved results** (CSV/TXT + `y_* .npy`) to produce:

  * Split size and positive-rate plots.
  * RF/XGB sweep visualizations (AP_val vs AP_test).
  * Barplots comparing RF / XGB / CNN / GNN on OOD AP & ROC-AUC.
  * Optional precision–recall curves for a chosen model.

These figures are designed to drop directly into the final report.

---

## Limitations & future extensions

The repository implements the **baseline pipeline**. Some planned extensions are **not** included here (on purpose):

* No hybrid model combining ECFP fingerprints and neural embeddings.
* No explicit probability calibration (Platt, isotonic) or decision-threshold selection.
* No model ensembling across RF/XGB/CNN/GNN.
* Experiments are limited to a 50k subset and moderate compute (single machine, no large sweeps).

Reasonable next steps:

* Add a hybrid ECFP + embedding model.
* Add post-hoc calibration and cost-sensitive threshold selection.
* Explore transformer models on SMILES or graphs.
* Compare block-based OOD splitting against scaffold-based or UMAP-based clustering splits.
* Incorporate uncertainty estimates and simple active-learning loops.

---

## License

This project is licensed under the **MIT License**.
See [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

* Course: **Machine Learning & Communication Skills @ IE University**.
* Dataset: BELKA BRD4 data from Kaggle (DNA-encoded library).
* Libraries: NumPy, pandas, scikit-learn, XGBoost, PyTorch, PyTorch Geometric, RDKit, and others.

```
```
```markdown
---

## Citation

If you use this repository or its components in your research, please cite:

```
@misc{fit_proteins2023,
    author = {Your Name},
    title = {F.I.T Proteins: Fast Inference of Therapeutic PROTEIN–Ligand Interactions for BRD4},
    year = {2023},
    url = {https://github.com/your-repo-url}
}
```

---

## Support

For questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/your-repo-url) or contact the maintainers directly.

---

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, please fork the repository, create a new branch, and submit a pull request. Make sure to follow the existing code style and include relevant tests where applicable.
```