import os

# Keep OpenMP/BLAS single-threaded before importing heavy ML libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import random
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

RANDOM_STATE = 42


def _configure_runtime_stability() -> None:
    # Mixed OpenMP stacks (xgboost, sklearn, torch) can crash on macOS when fully parallel.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads can only be set once per process.
        pass


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    _configure_runtime_stability()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FraudLSTM(nn.Module):
    def __init__(self, input_size: int = 30, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step).squeeze(-1)
        return logits


def create_padded_sequences(X: np.ndarray, y: np.ndarray | None = None, seq_len: int = 10):
    n_samples, n_features = X.shape
    sequences = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)

    for i in range(n_samples):
        start = max(0, i - seq_len + 1)
        window = X[start : i + 1]
        if window.shape[0] < seq_len:
            pad = np.repeat(window[[0]], seq_len - window.shape[0], axis=0)
            window = np.vstack([pad, window])
        sequences[i] = window

    if y is None:
        return sequences

    return sequences, y.astype(np.float32)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    epochs: int = 20,
    seq_len: int = 10,
    lr: float = 1e-3,
    batch_size: int = 512,
):
    set_global_seed(RANDOM_STATE)

    X_train_seq, y_train_seq = create_padded_sequences(X_train, y_train, seq_len=seq_len)
    X_val_seq, y_val_seq = create_padded_sequences(X_val, y_val, seq_len=seq_len)

    train_ds = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = FraudLSTM(input_size=X_train.shape[1], hidden_size=64, num_layers=2)

    n_fraud = max(int((y_train == 1).sum()), 1)
    n_genuine = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor(float(n_genuine / n_fraud), dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auprc = -np.inf
    best_state = None
    history = {"epoch": [], "train_loss": [], "val_auprc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val_seq, dtype=torch.float32))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auprc = average_precision_score(y_val_seq, val_probs)

        avg_train_loss = float(np.mean(losses)) if losses else 0.0
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_auprc"].append(float(val_auprc))

        if val_auprc > best_val_auprc:
            best_val_auprc = float(val_auprc)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seq_len": seq_len,
            "input_size": X_train.shape[1],
            "hidden_size": 64,
            "num_layers": 2,
        },
        model_path,
    )

    return model, history


def predict_lstm_proba(model: FraudLSTM, X: np.ndarray, seq_len: int = 10, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    X_seq = create_padded_sequences(X, y=None, seq_len=seq_len)

    probs = []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            batch = torch.tensor(X_seq[i : i + batch_size], dtype=torch.float32)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs, axis=0)


def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_dir: str = "models",
) -> Dict[str, object]:
    set_global_seed(RANDOM_STATE)
    os.makedirs(model_dir, exist_ok=True)

    y_train_arr = y_train.to_numpy(dtype=int)
    y_val_arr = y_val.to_numpy(dtype=int)

    logreg = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    logreg.fit(X_train, y_train_arr)
    joblib.dump(logreg, os.path.join(model_dir, "logreg.pkl"))

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf.fit(X_train, y_train_arr)
    joblib.dump(rf, os.path.join(model_dir, "rf.pkl"))

    n_fraud = max(int((y_train_arr == 1).sum()), 1)
    n_genuine = max(int((y_train_arr == 0).sum()), 1)
    scale_pos_weight = float(n_genuine / n_fraud)

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train_arr)
    joblib.dump(xgb, os.path.join(model_dir, "xgb.pkl"))

    lstm_model, lstm_history = train_lstm(
        X_train=X_train.to_numpy(dtype=np.float32),
        y_train=y_train_arr,
        X_val=X_val.to_numpy(dtype=np.float32),
        y_val=y_val_arr,
        model_path=os.path.join(model_dir, "lstm.pt"),
        epochs=20,
        seq_len=10,
        lr=1e-3,
    )

    return {
        "logreg": logreg,
        "rf": rf,
        "xgb": xgb,
        "lstm": lstm_model,
        "lstm_seq_len": 10,
        "lstm_history": lstm_history,
    }
