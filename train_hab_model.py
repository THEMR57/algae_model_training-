#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
    if non_constant_idx.size >= 2:
        corr_non_constant = np.corrcoef(sample_x[:, non_constant], rowvar=False)
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
    inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    inv_sqrt[deg <= 0] = 0.0
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

    df = pd.read_csv(args.data_path)
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

    test_m = run_epoch(model, test_loader, adj, criterion, optimizer, device, train=False)
    print(
        "test_metrics "
        f"loss={test_m.loss:.4f} accuracy={test_m.accuracy:.4f} "
        f"precision={test_m.precision:.4f} recall={test_m.recall:.4f} f1={test_m.f1:.4f}"
    )

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
            "test_metrics": test_m.__dict__,
        },
        checkpoint_path,
    )
    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"saved_model={checkpoint_path}")


if __name__ == "__main__":
    main()
