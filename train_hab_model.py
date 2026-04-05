#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NODE_EMBED_INIT_STD = 0.02
CONSTANT_FEATURE_STD_THRESHOLD = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels length mismatch")
        if len(features) <= seq_len:
            raise ValueError("dataset too small for selected seq_len")
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    valid_cols = []
    for col in df.columns:
        if col == "" or col.lower().startswith("unnamed:"):
            continue
        if df[col].isna().all():
            continue
        valid_cols.append(col)
    return df[valid_cols]


def build_target(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    if "target_hab_label" in df.columns:
        y = pd.to_numeric(df["target_hab_label"], errors="coerce").fillna(0).astype(int).to_numpy()
        return y, "target_hab_label"
    if "target_hab_probability" in df.columns:
        prob = pd.to_numeric(df["target_hab_probability"], errors="coerce").fillna(0).to_numpy()
        return (prob >= 0.5).astype(int), "target_hab_probability>=0.5"
    raise ValueError("No target column found. Expected target_hab_label or target_hab_probability.")


def prepare_features(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, List[str]]:
    feature_df = df.drop(columns=[c for c in ["target_hab_label", "target_hab_probability"] if c in df.columns])
    if "system:index" in feature_df.columns:
        feature_df = feature_df.drop(columns=["system:index"])
    numeric = feature_df.apply(pd.to_numeric, errors="coerce")
    # First fill with per-column medians; second fill handles columns that are entirely NaN.
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    cols = [c for c in numeric.columns if c != target_col]
    if not cols:
        raise ValueError("No usable numeric feature columns found.")
    x = numeric[cols].to_numpy(dtype=np.float32)
    return x, cols


def standardize(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def split_data(x: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, ...]:
    n = len(x)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid split ratios for dataset size.")
    return (
        x[:train_end],
        y[:train_end],
        x[train_end:val_end],
        y[train_end:val_end],
        x[val_end:],
        y[val_end:],
    )


def build_feature_adjacency(train_x: np.ndarray, top_k: int, max_rows: int = 50000, seed: int = 42) -> torch.Tensor:
    """Build a correlation-based normalized feature graph.

    Uses absolute Pearson correlation between feature columns, keeps top_k
    strongest neighbors per feature, symmetrizes edges, adds self-loops, and
    applies symmetric degree normalization. If row count exceeds max_rows, a
    deterministic random sample is used for faster correlation estimation.
    """
    if len(train_x) > max_rows:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(train_x), size=max_rows, replace=False)
        sample_x = train_x[sample_idx]
    else:
        sample_x = train_x
    n = sample_x.shape[1]
    corr = np.zeros((n, n), dtype=np.float64)
    non_constant = sample_x.std(axis=0) > CONSTANT_FEATURE_STD_THRESHOLD
    non_constant_idx = np.flatnonzero(non_constant)
    # Need at least two non-constant features to estimate pairwise correlation.
    if non_constant_idx.size >= 2:
        corr_non_constant = np.corrcoef(sample_x[:, non_constant_idx], rowvar=False)
        # Keep as numeric safety fallback for pathological finite-precision cases.
        corr[np.ix_(non_constant_idx, non_constant_idx)] = np.nan_to_num(corr_non_constant, nan=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    adj = np.zeros_like(corr)
    for i in range(n):
        idx = np.argsort(corr[i])[-top_k:]
        adj[i, idx] = corr[i, idx]
    adj = np.maximum(adj, adj.T)
    adj += np.eye(n, dtype=np.float64)
    deg = adj.sum(axis=1)
    # Use explicit `out` buffer to avoid uninitialized values from masked power op.
    inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=inv_sqrt)
    norm = (inv_sqrt[:, None] * adj) * inv_sqrt[None, :]
    return torch.tensor(norm, dtype=torch.float32)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        self_part = self.self_linear(h)
        # i,j are feature nodes; b=batch, t=time, d=embedding dimension.
        neigh = torch.einsum("ij,btjd->btid", adj, h)
        neigh_part = self.neigh_linear(neigh)
        return self.activation(self_part + neigh_part)


class STGNNTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        gnn_dim: int = 64,
        transformer_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.value_proj = nn.Linear(1, gnn_dim)
        self.node_embed = nn.Parameter(torch.randn(num_features, gnn_dim) * NODE_EMBED_INIT_STD)
        self.gnn = GraphConv(gnn_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        if f != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {f}")
        x = x.unsqueeze(-1)
        h = self.value_proj(x) + self.node_embed.view(1, 1, f, -1)
        h = self.gnn(h, adj)
        temporal = h.mean(dim=2)
        temporal = self.transformer(temporal)
        out = self.norm(temporal[:, -1, :])
        logits = self.head(out).squeeze(-1)
        return logits


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    labels = labels.float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = 0.5 * (recall + specificity)
    try:
        roc_auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except ValueError:
        pr_auc = float("nan")
    mcc = float(matthews_corrcoef(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0
    kappa = float(cohen_kappa_score(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(np.mean((probs - y_true) ** 2))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "kappa": kappa,
        "brier_score": brier,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y_true, (probs >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def collect_probs_labels(
    model: nn.Module, loader: DataLoader, adj: torch.Tensor, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x, adj)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(y.numpy().astype(int))
    return np.concatenate(probs_all), np.concatenate(labels_all)


def build_engineered_sequence_features(features: np.ndarray, labels: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(features) <= seq_len:
        raise ValueError("Dataset too small for engineered sequence features.")
    # Match SequenceDataset alignment: input window [i : i+seq_len) predicts label at i+seq_len.
    windows = np.lib.stride_tricks.sliding_window_view(features[:-1], window_shape=seq_len, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    mean_feat = windows.mean(axis=1)
    std_feat = windows.std(axis=1)
    min_feat = windows.min(axis=1)
    max_feat = windows.max(axis=1)
    last_feat = windows[:, -1, :]
    engineered = np.concatenate([mean_feat, std_feat, min_feat, max_feat, last_feat], axis=1).astype(np.float32)
    seq_labels = labels[seq_len:].astype(int)
    if len(engineered) != len(seq_labels):
        raise ValueError("Engineered sequence feature length mismatch.")
    return engineered, seq_labels


def optimize_ensemble(
    y_val: np.ndarray,
    deep_val_probs: np.ndarray,
    tree_val_probs: np.ndarray,
) -> Tuple[float, float, float]:
    best_weight = 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for weight in np.linspace(0.0, 1.0, 41):
        probs = weight * deep_val_probs + (1.0 - weight) * tree_val_probs
        threshold, f1 = find_best_threshold(y_val, probs)
        if f1 > best_f1:
            best_f1 = f1
            best_weight = float(weight)
            best_threshold = float(threshold)
    return best_weight, best_threshold, best_f1


def save_training_curves(history: List[Dict[str, float]], output_dir: str) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h["train_f1"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[1].plot(epochs, train_f1, label="Train F1")
    axes[1].plot(epochs, val_f1, label="Val F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 Curve")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=180)
    plt.close(fig)


def save_roc_pr_curves(y_true: np.ndarray, probs: np.ndarray, output_dir: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    fig1 = plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=180)
    plt.close(fig1)
    fig2 = plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=180)
    plt.close(fig2)


def save_confusion_matrix_plot(y_true: np.ndarray, probs: np.ndarray, threshold: float, output_dir: str) -> None:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=180)
    plt.close(fig)


def save_probability_distribution(y_true: np.ndarray, probs: np.ndarray, output_dir: str) -> None:
    fig = plt.figure(figsize=(6, 4))
    plt.hist(probs[y_true == 0], bins=40, alpha=0.6, label="Negative Class")
    plt.hist(probs[y_true == 1], bins=40, alpha=0.6, label="Positive Class")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "probability_distribution.png"), dpi=180)
    plt.close(fig)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    adj: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> EpochMetrics:
    model.train(train)
    losses: List[float] = []
    logits_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x, adj)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(loss.item())
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())
    if not losses:
        return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    met = binary_metrics(logits_cat, labels_cat)
    return EpochMetrics(
        loss=float(np.mean(losses)),
        accuracy=met["accuracy"],
        precision=met["precision"],
        recall=met["recall"],
        f1=met["f1"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid GNN + Transformer HAB prediction model.")
    parser.add_argument("--data-path", type=str, default="hab_dataset_200000_rows.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--top-k-neighbors", type=int, default=4)
    parser.add_argument("--hgb-max-iter", type=int, default=300)
    parser.add_argument("--hgb-max-depth", type=int, default=8)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # Jupyter/Colab kernels inject `-f <connection_file>`; accept it silently.
    parser.add_argument("-f", dest="_ipykernel_connection_file", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path, low_memory=False)
    df = clean_dataframe(df)
    if "system:index" in df.columns:
        df = df.sort_values("system:index").reset_index(drop=True)
    y, target_origin = build_target(df)
    x, feature_cols = prepare_features(df, target_origin)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y, args.train_ratio, args.val_ratio)
    x_train, x_val, x_test, mean, std = standardize(x_train, x_val, x_test)

    train_ds = SequenceDataset(x_train, y_train, args.seq_len)
    val_ds = SequenceDataset(x_val, y_val, args.seq_len)
    test_ds = SequenceDataset(x_test, y_test, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    effective_top_k = max(1, min(args.top_k_neighbors, x_train.shape[1] - 1))
    adj = build_feature_adjacency(
        x_train,
        top_k=effective_top_k,
        seed=args.seed,
    ).to(device)

    model = STGNNTransformer(num_features=x_train.shape[1]).to(device)

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - y_train.sum())
    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            f"Training split must contain both classes. Got positives={pos_count}, negatives={neg_count}."
        )
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_f1 = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, adj, criterion, optimizer, device, train=True)
        val_m = run_epoch(model, val_loader, adj, criterion, optimizer, device, train=False)
        scheduler.step(val_m.f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_m.loss,
                "train_f1": train_m.f1,
                "val_loss": val_m.loss,
                "val_f1": val_m.f1,
            }
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_m.loss:.4f} train_f1={train_m.f1:.4f} "
            f"val_loss={val_m.loss:.4f} val_f1={val_m.f1:.4f}"
        )
        if val_m.f1 > best_f1:
            best_f1 = val_m.f1
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Deep model probabilities
    val_deep_probs, val_deep_labels = collect_probs_labels(model, val_loader, adj, device)
    test_deep_probs, test_deep_labels = collect_probs_labels(model, test_loader, adj, device)

    # Sequence-engineered tabular model
    train_tab_x, train_tab_y = build_engineered_sequence_features(x_train, y_train, args.seq_len)
    val_tab_x, val_tab_y = build_engineered_sequence_features(x_val, y_val, args.seq_len)
    test_tab_x, test_tab_y = build_engineered_sequence_features(x_test, y_test, args.seq_len)
    if not (np.array_equal(val_deep_labels, val_tab_y) and np.array_equal(test_deep_labels, test_tab_y)):
        raise ValueError(
            "Label alignment mismatch between deep and tabular sequence pipelines. "
            f"Val: {len(val_deep_labels)} vs {len(val_tab_y)}, "
            f"Test: {len(test_deep_labels)} vs {len(test_tab_y)}."
        )

    tab_model = HistGradientBoostingClassifier(
        learning_rate=args.hgb_learning_rate,
        max_depth=args.hgb_max_depth,
        max_iter=args.hgb_max_iter,
        min_samples_leaf=32,
        l2_regularization=1e-3,
        validation_fraction=0.1,
        early_stopping=True,
        random_state=args.seed,
    )
    tab_model.fit(train_tab_x, train_tab_y)
    val_tab_probs = tab_model.predict_proba(val_tab_x)[:, 1]
    test_tab_probs = tab_model.predict_proba(test_tab_x)[:, 1]

    deep_threshold, deep_val_f1 = find_best_threshold(val_deep_labels, val_deep_probs)
    tab_threshold, tab_val_f1 = find_best_threshold(val_tab_y, val_tab_probs)
    ensemble_weight, ensemble_threshold, ensemble_val_f1 = optimize_ensemble(val_tab_y, val_deep_probs, val_tab_probs)

    deep_test_metrics = compute_metrics_from_probs(test_deep_labels, test_deep_probs, deep_threshold)
    tab_test_metrics = compute_metrics_from_probs(test_tab_y, test_tab_probs, tab_threshold)
    ensemble_test_probs = ensemble_weight * test_deep_probs + (1.0 - ensemble_weight) * test_tab_probs
    ensemble_test_metrics = compute_metrics_from_probs(test_tab_y, ensemble_test_probs, ensemble_threshold)

    model_records = [
        {
            "model": "stgnn_transformer",
            "val_best_f1": deep_val_f1,
            **deep_test_metrics,
        },
        {
            "model": "hist_gradient_boosting",
            "val_best_f1": tab_val_f1,
            **tab_test_metrics,
        },
        {
            "model": "hybrid_ensemble",
            "val_best_f1": ensemble_val_f1,
            "ensemble_weight_deep": ensemble_weight,
            "ensemble_weight_tabular": 1.0 - ensemble_weight,
            **ensemble_test_metrics,
        },
    ]

    selected = max(
        [
            ("stgnn_transformer", deep_val_f1, test_deep_probs, deep_threshold, deep_test_metrics),
            ("hist_gradient_boosting", tab_val_f1, test_tab_probs, tab_threshold, tab_test_metrics),
            ("hybrid_ensemble", ensemble_val_f1, ensemble_test_probs, ensemble_threshold, ensemble_test_metrics),
        ],
        key=lambda item: item[1],
    )
    selected_name, selected_val_f1, selected_probs, selected_threshold, selected_metrics = selected
    print(
        f"selected_model={selected_name} selected_val_f1={selected_val_f1:.4f} "
        f"test_accuracy={selected_metrics['accuracy']:.4f} test_f1={selected_metrics['f1']:.4f}"
    )

    pd.DataFrame(model_records).to_csv(os.path.join(args.output_dir, "performance_metrics_matrix.csv"), index=False)
    with open(os.path.join(args.output_dir, "performance_metrics_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(model_records, f, indent=2)
    with open(os.path.join(args.output_dir, "best_model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": selected_name,
                "selected_model_val_f1": selected_val_f1,
                "selected_threshold": selected_threshold,
                "selected_metrics": selected_metrics,
                "ensemble_weight_deep": ensemble_weight,
                "ensemble_weight_tabular": 1.0 - ensemble_weight,
            },
            f,
            indent=2,
        )

    save_training_curves(history, args.output_dir)
    save_roc_pr_curves(test_tab_y, selected_probs, args.output_dir)
    save_confusion_matrix_plot(test_tab_y, selected_probs, selected_threshold, args.output_dir)
    save_probability_distribution(test_tab_y, selected_probs, args.output_dir)

    checkpoint_path = os.path.join(args.output_dir, "stgnn_transformer_hab.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "adjacency": adj.detach().cpu(),
            "feature_columns": feature_cols,
            "feature_mean": mean,
            "feature_std": std,
            "config": vars(args),
            "target_definition": target_origin,
            "best_val_f1": best_f1,
            "model_comparison": model_records,
            "selected_model": selected_name,
            "selected_metrics": selected_metrics,
            "selected_threshold": selected_threshold,
            "ensemble_weight_deep": ensemble_weight,
            "ensemble_weight_tabular": 1.0 - ensemble_weight,
        },
        checkpoint_path,
    )
    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"saved_model={checkpoint_path}")


if __name__ == "__main__":
    main()
