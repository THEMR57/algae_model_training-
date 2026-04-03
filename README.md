# algae_model_training-

Train a Harmful Algal Bloom (HAB) model using a hybrid **Graph Neural Network + Transformer** architecture on `hab_dataset_200000_rows.csv`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train_hab_model.py \
  --data-path hab_dataset_200000_rows.csv \
  --output-dir artifacts \
  --epochs 25 \
  --batch-size 256 \
  --seq-len 24
```

## Outputs

Training writes:

- `artifacts/stgnn_transformer_hab.pt` (model checkpoint + preprocessing metadata)
- `artifacts/training_history.json` (epoch-wise train/validation history)
