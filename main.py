import os

# Keep OpenMP/BLAS single-threaded before any heavy imports on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import random
import time

import joblib
import numpy as np
import pandas as pd
import torch

from src.baselines import predict_lstm_proba, train_baselines
from src.drift_module import BehavioralDriftDetector
from src.evaluate import compute_metrics, generate_all_graphs, print_metrics_table
from src.fusion import classify, fuse_scores, tune_fusion_weights
from src.graph_module import build_graph_context, score_dataframe_with_graph, train_gnn
from src.predict import predict_single
from src.preprocessing import FEATURE_COLUMNS, load_and_preprocess

RANDOM_STATE = 42


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_project_dirs() -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/graphs", exist_ok=True)


def _inference_timing_ms(start_time: float, n_samples: int) -> float:
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return elapsed_ms / max(n_samples, 1)


def _evaluate_baselines(baseline_models, X_test, y_test):
    model_outputs = {}
    metric_rows = []

    # Logistic Regression
    start = time.perf_counter()
    logreg_prob = baseline_models["logreg"].predict_proba(X_test)[:, 1]
    logreg_ms = _inference_timing_ms(start, len(X_test))
    logreg_pred = (logreg_prob >= 0.5).astype(int)
    metric_rows.append(compute_metrics(y_test, logreg_pred, logreg_prob, "Logistic Reg", inference_ms=logreg_ms))
    model_outputs["Logistic Reg"] = {"y_true": y_test.to_numpy(), "y_prob": logreg_prob, "y_pred": logreg_pred}

    # Random Forest
    start = time.perf_counter()
    rf_prob = baseline_models["rf"].predict_proba(X_test)[:, 1]
    rf_ms = _inference_timing_ms(start, len(X_test))
    rf_pred = (rf_prob >= 0.5).astype(int)
    metric_rows.append(compute_metrics(y_test, rf_pred, rf_prob, "Random Forest", inference_ms=rf_ms))
    model_outputs["Random Forest"] = {"y_true": y_test.to_numpy(), "y_prob": rf_prob, "y_pred": rf_pred}

    # XGBoost
    start = time.perf_counter()
    xgb_prob = baseline_models["xgb"].predict_proba(X_test)[:, 1]
    xgb_ms = _inference_timing_ms(start, len(X_test))
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    metric_rows.append(compute_metrics(y_test, xgb_pred, xgb_prob, "XGBoost", inference_ms=xgb_ms))
    model_outputs["XGBoost"] = {"y_true": y_test.to_numpy(), "y_prob": xgb_prob, "y_pred": xgb_pred}

    # LSTM
    start = time.perf_counter()
    lstm_prob = predict_lstm_proba(
        baseline_models["lstm"],
        X_test.to_numpy(dtype=np.float32),
        seq_len=baseline_models.get("lstm_seq_len", 10),
    )
    lstm_ms = _inference_timing_ms(start, len(X_test))
    lstm_pred = (lstm_prob >= 0.5).astype(int)
    metric_rows.append(compute_metrics(y_test, lstm_pred, lstm_prob, "LSTM", inference_ms=lstm_ms))
    model_outputs["LSTM"] = {"y_true": y_test.to_numpy(), "y_prob": lstm_prob, "y_pred": lstm_pred}

    return metric_rows, model_outputs


def main() -> None:
    set_global_seed(RANDOM_STATE)
    ensure_project_dirs()

    print("\n[1/12] Loading and preprocessing data...")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        X_train_raw,
        y_train_raw,
    ) = load_and_preprocess(path="data/creditcard.csv", return_raw=True)

    joblib.dump(scaler, "models/scaler.pkl")

    print("[2/12] Training baseline models (LogReg, RF, XGBoost, LSTM)...")
    baseline_models = train_baselines(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_dir="models",
    )

    print("[3/12] Training behavioral drift detector...")
    drift_detector = BehavioralDriftDetector(window=50)
    drift_detector.fit(X_train_raw[["Time", "Amount"]])
    joblib.dump(drift_detector, "models/drift_detector.pkl")

    print("[4/12] Computing drift scores for train/val/test...")
    train_drift = drift_detector.compute_drift_score(X_train_raw[["Time", "Amount"]])
    val_drift = drift_detector.compute_drift_score(X_val[["Time", "Amount"]])
    test_drift = drift_detector.compute_drift_score(X_test[["Time", "Amount"]])

    print("[5/12] Building transaction graph and training GNN...")
    gnn_model, _, gnn_history = train_gnn(
        X_train=X_train_raw[FEATURE_COLUMNS],
        y_train=y_train_raw,
        X_val=X_val[FEATURE_COLUMNS],
        y_val=y_val,
        model_path="models/gnn_best.pt",
        epochs=50,
        patience=10,
        lr=1e-3,
        weight_decay=1e-4,
    )

    print("[6/12] Computing graph anomaly scores for train/val/test...")
    train_graph_scores = score_dataframe_with_graph(gnn_model, X_train_raw[FEATURE_COLUMNS], k=5)
    val_graph_scores = score_dataframe_with_graph(gnn_model, X_val[FEATURE_COLUMNS], k=5)
    test_graph_scores = score_dataframe_with_graph(gnn_model, X_test[FEATURE_COLUMNS], k=5)

    print("[7/12] Tuning fusion weights and threshold on validation set...")
    drift_bounds = (float(np.min(train_drift)), float(np.max(train_drift)))
    graph_bounds = (float(np.min(train_graph_scores)), float(np.max(train_graph_scores)))

    best_fusion = tune_fusion_weights(
        drift_val=val_drift,
        graph_val=val_graph_scores,
        y_val=y_val,
        drift_bounds=drift_bounds,
        graph_bounds=graph_bounds,
    )

    w_drift = float(best_fusion["w_drift"])
    w_graph = float(best_fusion["w_graph"])
    optimal_threshold = float(best_fusion["threshold"])

    print("[8/12] Running baseline model inference on test set...")
    metric_rows, model_outputs = _evaluate_baselines(baseline_models, X_test, y_test)

    print("[9/12] Computing ABD-GAF fusion predictions on test set...")
    start = time.perf_counter()
    fusion_test_scores = fuse_scores(
        drift_scores=test_drift,
        graph_scores=test_graph_scores,
        w_drift=w_drift,
        w_graph=w_graph,
        drift_bounds=drift_bounds,
        graph_bounds=graph_bounds,
    )
    fusion_ms = _inference_timing_ms(start, len(X_test))

    abd_pred = classify(fusion_test_scores, threshold=optimal_threshold)
    abd_metrics = compute_metrics(
        y_true=y_test,
        y_pred=abd_pred,
        y_prob=fusion_test_scores,
        model_name="ABD-GAF (proposed)",
        inference_ms=fusion_ms,
    )
    metric_rows.append(abd_metrics)

    model_outputs["ABD-GAF (proposed)"] = {
        "y_true": y_test.to_numpy(),
        "y_prob": fusion_test_scores,
        "y_pred": abd_pred,
    }

    print("[10/12] Generating metrics table and benchmark CSV...")
    metrics_df = pd.DataFrame(metric_rows)
    model_order = ["Logistic Reg", "Random Forest", "XGBoost", "LSTM", "ABD-GAF (proposed)"]
    metrics_df["model"] = pd.Categorical(metrics_df["model"], categories=model_order, ordered=True)
    metrics_df = metrics_df.sort_values("model").reset_index(drop=True)

    print_metrics_table(metrics_df)
    metrics_df.to_csv("results/benchmark_scores.csv", index=False)

    print("[11/12] Generating and saving all required graphs...")
    results_dict = {
        "model_outputs": model_outputs,
        "metrics_df": metrics_df,
        "abd_gaf": {
            "y_true": y_test.to_numpy(),
            "y_pred": abd_pred,
            "drift_scores": test_drift,
            "graph_scores": test_graph_scores,
            "fusion_scores": fusion_test_scores,
            "drift_threshold": drift_detector.drift_threshold,
            "optimal_threshold": optimal_threshold,
        },
        "gnn_history": gnn_history,
    }
    generate_all_graphs(results_dict=results_dict, test_data={"y_test": y_test.to_numpy()})

    graph_context = build_graph_context(
        X_train=X_train_raw[FEATURE_COLUMNS],
        train_graph_scores=train_graph_scores,
        drift_bounds=drift_bounds,
        graph_bounds=graph_bounds,
        w_drift=w_drift,
        w_graph=w_graph,
    )
    joblib.dump(graph_context, "models/graph_context.pkl")
    joblib.dump(
        {
            "optimal_threshold": optimal_threshold,
            "w_drift": w_drift,
            "w_graph": w_graph,
            "val_auprc": float(best_fusion["auprc"]),
        },
        "models/fusion_config.pkl",
    )

    print("[12/12] Running demo single prediction using a real fraud sample from test split...")
    fraud_candidates = X_test[y_test == 1]
    if len(fraud_candidates) == 0:
        print("No fraud sample found in test split for demo prediction.")
    else:
        sample_fraud = fraud_candidates.iloc[0]
        _ = predict_single(
            transaction=sample_fraud.to_dict(),
            scaler=scaler,
            drift_detector=drift_detector,
            gnn_model=gnn_model,
            graph_data=graph_context,
            threshold=optimal_threshold,
        )

    print("\nPipeline complete. Artifacts saved under models/ and results/.")


if __name__ == "__main__":
    main()
