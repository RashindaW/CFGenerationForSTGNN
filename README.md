# CFGenerationForSTGNN

Utilities for experimenting with counterfactual generation on spatio-temporal GNNs.  
This repository now provides modular model definitions, preprocessing helpers, and a single training entry point.

## Getting Started

```bash
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-geometric
```

## Training

```bash
python train.py --mode train --model stgcn --dataset METRLA
```

- Splits the dataset into 70 % train, 10 % validation, 20 % test.
- Tracks validation loss and saves the best checkpoint to `checkpoints/<timestamp>_<model>_<dataset>/best.pt` by default.
- Override the destination with `--checkpoint_dir` or a full `--output` path if you want to name the file yourself.
- Tune hyperparameters on the fly; e.g. `--epochs 100`, `--batch_size 32`, `--learning_rate 3e-4`, `--weight_decay 1e-5`, `--patience 20`, `--grad_clip 2.0`.

## Testing

```bash
python train.py --mode test --checkpoint checkpoints/<timestamp>_<model>_<dataset>/best.pt
```

- Rebuilds the model using the saved training arguments and evaluates only on the stored test split.
- Add flags such as `--data_root` if you need to point at a different dataset location than was captured in the checkpoint.
- You can still override evaluation settings (e.g. `--batch_size 128`) when re-running in test mode.

## Useful Flags

- `--model`: `stgcn` or `graphwavenet`
- `--dataset`: `METRLA` or `PEMSBAY`
- `--lag` / `--horizon`: historical context and forecast length
- `--checkpoint_dir`: directory where training runs are stored (defaults to `checkpoints`)
- `--checkpoint`: path to a saved checkpoint when running in `test` mode
- `--output`: explicit checkpoint filepath; overrides the directory layout when provided

Datasets are expected under `preprocessing/data/<DATASET>` with `*_node_values.npy` and `*_adj_mat.npy` files. 
