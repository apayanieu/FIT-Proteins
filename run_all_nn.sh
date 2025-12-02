#!/usr/bin/env bash
set -e  # stop on first error

# 1) SMILES → tokenized arrays for CNN
python scripts/smiles_nn_token.py prepare \
  --label-col binds \
  --max-len 256

# 2) SMILES → PyG graphs for GNN
python scripts/gnn_pyg_gen.py prepare \
  --smiles-col smiles_clean \
  --label-col binds

# 3) Train SMILES-CNN
python src/fit_proteins/models/smiles_cnn.py train_cnn \
  --data-dir data/processed \
  --results-dir results/cnn_results

# 4) Train GNN
python src/fit_proteins/models/gnn.py train_gnn \
  --data-dir data/processed \
  --results-dir results/gnn_results
