# algae_model_training-

Train a Harmful Algal Bloom (HAB) model using a research-focused **hybrid ensemble** pipeline:

- **STGNN + Transformer** deep sequence model
- **Histogram Gradient Boosting** tabular sequence-statistics model
- **Weighted probability ensemble** with validation-time threshold tuning

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
  --seq-len 24 \
  --hgb-max-iter 300 \
  --hgb-max-depth 8 \
  --hgb-learning-rate 0.05
```

## Outputs

Training writes:

- `artifacts/stgnn_transformer_hab.pt` (model checkpoint + preprocessing metadata)
- `artifacts/training_history.json` (epoch-wise train/validation history)
- `artifacts/performance_metrics_matrix.csv` and `.json` (multi-model performance matrix)
- `artifacts/best_model_summary.json` (selected best model + tuned threshold)

Graphs written for research reporting:

- `artifacts/training_curves.png` (train/validation loss + F1)
- `artifacts/roc_curve.png`
- `artifacts/precision_recall_curve.png`
- `artifacts/confusion_matrix.png`
- `artifacts/probability_distribution.png`

## Performance matrix metrics

The performance matrix includes multiple metrics for each candidate model:

- Accuracy, Precision, Recall, Specificity, F1
- Balanced Accuracy, ROC-AUC, PR-AUC
- MCC, Cohen’s Kappa, Brier Score
- Confusion matrix counts (TP, TN, FP, FN)

The final selected model is the one with best validation F1 (after threshold tuning), intended to improve practical classification performance versus a single-model baseline.
