# CFGenerationForSTGNN

Utilities for experimenting with counterfactual generation on spatio-temporal GNNs.  
This repository now provides modular model definitions, preprocessing helpers, and a single training entry point.

## Getting Started

```bash
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-geometric
python train.py --model stgcn --dataset METRLA
```

Key flags:

- `--model`: `stgcn` or `graphwavenet`
- `--dataset`: `METRLA` or `PEMSBAY`
- `--lag` / `--horizon`: historical context and forecast length
- `--output`: optional path to persist trained weights

Datasets are expected under `preprocessing/data/<DATASET>` with `*_node_values.npy` and `*_adj_mat.npy` files. 