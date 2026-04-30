import os

# Keep OpenMP/BLAS single-threaded before importing heavy ML libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

RANDOM_STATE = 42


def _configure_runtime_stability() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    _configure_runtime_stability()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_knn_graph(feature_matrix: np.ndarray, k: int = 5) -> Data:
    n_samples = feature_matrix.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot build graph from empty feature matrix.")

    if n_samples == 1:
        x = torch.tensor(feature_matrix, dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    n_neighbors = min(k + 1, n_samples)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=1)
    knn.fit(feature_matrix)
    distances, indices = knn.kneighbors(feature_matrix)

    src_nodes = []
    dst_nodes = []
    weights = []

    for src, neigh_list in enumerate(indices):
        for j, dst in enumerate(neigh_list):
            if src == dst:
                continue
            sim = 1.0 - distances[src, j]
            src_nodes.append(src)
            dst_nodes.append(int(dst))
            weights.append(float(sim))

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    x = torch.tensor(feature_matrix, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


class FraudGNN(nn.Module):
    def __init__(self, in_channels: int = 30, hidden_1: int = 64, hidden_2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_1)
        self.conv2 = SAGEConv(hidden_1, hidden_2)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        logits = self.out(x).squeeze(-1)
        return logits


def train_gnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_path: str = "models/gnn_best.pt",
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[FraudGNN, Data, Dict[str, list]]:
    set_global_seed(RANDOM_STATE)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X_full = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_full = pd.concat([y_train, y_val], axis=0, ignore_index=True).astype(float)

    data = build_knn_graph(X_full.to_numpy(dtype=np.float32), k=5)

    n_train = len(X_train)
    n_total = len(X_full)

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.y = torch.tensor(y_full.to_numpy(), dtype=torch.float32)

    model = FraudGNN(in_channels=X_full.shape[1])

    n_fraud = max(int((y_train == 1).sum()), 1)
    n_genuine = max(int((y_train == 0).sum()), 1)
    pos_weight = float(n_genuine / n_fraud)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"epoch": [], "train_loss": [], "val_auprc": []}
    best_val_auprc = -np.inf
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data)
        train_logits = logits[data.train_mask]
        train_targets = data.y[data.train_mask]

        loss = criterion(train_logits, train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(data)
            val_probs = torch.sigmoid(logits_eval[data.val_mask]).cpu().numpy()
            val_targets = data.y[data.val_mask].cpu().numpy()
            val_auprc = average_precision_score(val_targets, val_probs)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss.item()))
        history["val_auprc"].append(float(val_auprc))

        if val_auprc > best_val_auprc:
            best_val_auprc = float(val_auprc)
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            break

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, data, history


def compute_graph_score(model: FraudGNN, data: Data) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def score_dataframe_with_graph(model: FraudGNN, X: pd.DataFrame, k: int = 5) -> np.ndarray:
    data = build_knn_graph(X.to_numpy(dtype=np.float32), k=k)
    return compute_graph_score(model, data)


def build_graph_context(
    X_train: pd.DataFrame,
    train_graph_scores: np.ndarray,
    drift_bounds: Tuple[float, float],
    graph_bounds: Tuple[float, float],
    w_drift: float,
    w_graph: float,
) -> Dict[str, np.ndarray | list | tuple | float]:
    return {
        "train_features": X_train.to_numpy(dtype=np.float32),
        "feature_columns": list(X_train.columns),
        "train_graph_scores": np.asarray(train_graph_scores, dtype=np.float32),
        "drift_bounds": (float(drift_bounds[0]), float(drift_bounds[1])),
        "graph_bounds": (float(graph_bounds[0]), float(graph_bounds[1])),
        "w_drift": float(w_drift),
        "w_graph": float(w_graph),
    }
