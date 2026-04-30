from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.drift_module import BehavioralDriftDetector
from src.fusion import classify, fuse_scores

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
REQUIRED_KEYS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def _validate_transaction_dict(transaction: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in transaction]
    if missing:
        raise ValueError(f"Missing required transaction keys: {missing}")


def _to_feature_dataframe(transaction: dict) -> pd.DataFrame:
    _validate_transaction_dict(transaction)
    row = {col: float(transaction[col]) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a shape: (1, d), b shape: (n, d)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (a @ b.T) / (a_norm * b_norm.T)


def _nearest_neighbor_graph_score(feature_vector: np.ndarray, graph_data: dict) -> float:
    train_features = np.asarray(graph_data["train_features"], dtype=np.float32)
    train_graph_scores = np.asarray(graph_data["train_graph_scores"], dtype=np.float32)

    sims = _cosine_similarity(feature_vector.reshape(1, -1), train_features).ravel()
    nn_idx = int(np.argmax(sims))
    return float(train_graph_scores[nn_idx])


def _print_prediction_result(result: Dict[str, float | str]) -> None:
    verdict_text = "⚠ FRAUD" if result["verdict"] == "FRAUD" else "GENUINE"
    print("╔══════════════════════════════════════════╗")
    print("║       ABD-GAF FRAUD DETECTION RESULT     ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Verdict          : {verdict_text:<20}║")
    print(f"║  Fraud probability: {result['fraud_probability'] * 100:>5.1f}%{'':<14}║")
    print(f"║  Drift score      : {result['drift_score']:>5.2f}{'':<18}║")
    print(f"║  Graph score      : {result['graph_score']:>5.2f}{'':<18}║")
    print(f"║  Fusion score     : {result['fusion_score']:>5.2f}{'':<18}║")
    print(f"║  Threshold used   : {result['threshold_used']:>5.2f}{'':<18}║")
    print("╚══════════════════════════════════════════╝")


def predict_single(transaction: dict, scaler, drift_detector: BehavioralDriftDetector, gnn_model, graph_data, threshold) -> dict:
    """
    Predict fraud risk for a single transaction.

    Required transaction keys:
    Time, Amount, V1..V28
    """
    _ = gnn_model
    tx_df = _to_feature_dataframe(transaction)

    tx_df[["Time", "Amount"]] = scaler.transform(tx_df[["Time", "Amount"]])

    drift_score = float(drift_detector.compute_drift_score(tx_df[["Time", "Amount"]])[0])

    feature_vector = tx_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    graph_score = _nearest_neighbor_graph_score(feature_vector[0], graph_data)

    w_drift = float(graph_data.get("w_drift", 0.4))
    w_graph = float(graph_data.get("w_graph", 0.6))
    drift_bounds = graph_data.get("drift_bounds", None)
    graph_bounds = graph_data.get("graph_bounds", None)

    fusion_score = float(
        fuse_scores(
            np.array([drift_score]),
            np.array([graph_score]),
            w_drift=w_drift,
            w_graph=w_graph,
            drift_bounds=drift_bounds,
            graph_bounds=graph_bounds,
        )[0]
    )

    y_pred = int(classify(np.array([fusion_score]), threshold)[0])
    verdict = "FRAUD" if y_pred == 1 else "GENUINE"

    result = {
        "verdict": verdict,
        "fraud_probability": float(fusion_score),
        "drift_score": drift_score,
        "graph_score": float(graph_score),
        "fusion_score": float(fusion_score),
        "threshold_used": float(threshold),
    }

    _print_prediction_result(result)
    return result


def predict_batch(csv_path: str, scaler, drift_detector: BehavioralDriftDetector, gnn_model, graph_data, threshold) -> pd.DataFrame:
    """
    Batch inference for CSV files with creditcard.csv-compatible schema.
    """
    _ = gnn_model
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    out_df = df.copy()
    features_df = out_df[FEATURE_COLUMNS].copy()
    features_df[["Time", "Amount"]] = scaler.transform(features_df[["Time", "Amount"]])

    drift_scores = drift_detector.compute_drift_score(features_df[["Time", "Amount"]])

    train_features = np.asarray(graph_data["train_features"], dtype=np.float32)
    train_graph_scores = np.asarray(graph_data["train_graph_scores"], dtype=np.float32)

    all_features = features_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    sims = _cosine_similarity(all_features, train_features)
    nn_indices = np.argmax(sims, axis=1)
    graph_scores = train_graph_scores[nn_indices]

    w_drift = float(graph_data.get("w_drift", 0.4))
    w_graph = float(graph_data.get("w_graph", 0.6))
    drift_bounds = graph_data.get("drift_bounds", None)
    graph_bounds = graph_data.get("graph_bounds", None)

    fusion_scores = fuse_scores(
        drift_scores,
        graph_scores,
        w_drift=w_drift,
        w_graph=w_graph,
        drift_bounds=drift_bounds,
        graph_bounds=graph_bounds,
    )
    preds = classify(fusion_scores, threshold)

    out_df["predicted_label"] = preds
    out_df["fraud_probability"] = fusion_scores
    out_df["drift_score"] = drift_scores
    out_df["graph_score"] = graph_scores
    out_df["fusion_score"] = fusion_scores

    out_path = "results/batch_predictions.csv"
    out_df.to_csv(out_path, index=False)

    n_total = len(out_df)
    n_fraud = int((out_df["predicted_label"] == 1).sum())
    n_genuine = int((out_df["predicted_label"] == 0).sum())

    print(f"Batch prediction complete. Output saved to: {out_path}")
    print(f"Total transactions: {n_total}")
    print(f"Predicted frauds:   {n_fraud}")
    print(f"Predicted genuine:  {n_genuine}")

    return out_df
