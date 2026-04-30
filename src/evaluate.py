import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

MODEL_COLORS = {
    "Logistic Reg": "#888780",
    "Random Forest": "#1D9E75",
    "XGBoost": "#EF9F27",
    "LSTM": "#7F77DD",
    "ABD-GAF (proposed)": "#E24B4A",
}


def compute_metrics(y_true, y_pred, y_prob, model_name, inference_ms=None):
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    try:
        auc_roc = roc_auc_score(y_true_arr, y_prob_arr)
    except ValueError:
        auc_roc = np.nan

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(y_true_arr, y_pred_arr, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
        "auc_roc": auc_roc,
        "auprc": average_precision_score(y_true_arr, y_prob_arr),
        "inference_ms": float(inference_ms) if inference_ms is not None else np.nan,
    }
    return metrics


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    display_df = metrics_df.copy()
    display_df = display_df[
        ["model", "accuracy", "precision", "recall", "f1", "auc_roc", "auprc", "inference_ms"]
    ]

    print("\nModel              | Accuracy | Precision | Recall | F1    | AUC-ROC | AUPRC  | Time(ms)")
    print("-------------------|----------|-----------|--------|-------|---------|--------|----------")
    for _, row in display_df.iterrows():
        print(
            f"{row['model']:<19}| {row['accuracy']:.4f}   | {row['precision']:.3f}     "
            f"| {row['recall']:.3f}  | {row['f1']:.3f} | {row['auc_roc']:.3f}   "
            f"| {row['auprc']:.3f}  | {row['inference_ms']:.2f}"
        )


def _save_figure(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_graphs(results_dict: Dict, test_data: Dict):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    model_outputs = results_dict["model_outputs"]
    metrics_df = results_dict["metrics_df"]
    abd = results_dict["abd_gaf"]
    gnn_history = results_dict["gnn_history"]

    # Graph 1: ROC curves.
    plt.figure(figsize=(8, 6))
    for model_name, out in model_outputs.items():
        fpr, tpr, _ = roc_curve(out["y_true"], out["y_prob"])
        auc_val = roc_auc_score(out["y_true"], out["y_prob"])
        lw = 2.8 if model_name == "ABD-GAF (proposed)" else 1.8
        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={auc_val:.3f})",
            color=MODEL_COLORS.get(model_name, None),
            linewidth=lw,
        )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("ROC curve comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    _save_figure("results/graphs/01_roc_curves.png")

    # Graph 2: Precision-recall curves.
    plt.figure(figsize=(8, 6))
    for model_name, out in model_outputs.items():
        precision, recall, _ = precision_recall_curve(out["y_true"], out["y_prob"])
        auprc_val = average_precision_score(out["y_true"], out["y_prob"])
        lw = 2.8 if model_name == "ABD-GAF (proposed)" else 1.8
        plt.plot(
            recall,
            precision,
            label=f"{model_name} (AUPRC={auprc_val:.3f})",
            color=MODEL_COLORS.get(model_name, None),
            linewidth=lw,
        )
    plt.axhline(y=0.00172, color="black", linestyle="--", linewidth=1, label="No-skill baseline")
    plt.title("Precision-recall curve comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    _save_figure("results/graphs/02_pr_curves.png")

    # Graph 3: Confusion matrix for ABD-GAF.
    cm = confusion_matrix(abd["y_true"], abd["y_pred"])
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"],
    )
    plt.title("Confusion matrix - ABD-GAF (test set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _save_figure("results/graphs/03_confusion_matrix.png")

    # Graph 4: Grouped benchmark bars.
    plot_df = metrics_df[["model", "f1", "auprc"]].copy()
    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, plot_df["f1"], width=width, color="#1D9E75", label="F1")
    bars2 = plt.bar(x + width / 2, plot_df["auprc"], width=width, color="#E67E5F", label="AUPRC")

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.005, f"{height:.3f}", ha="center", va="bottom")

    plt.xticks(x, plot_df["model"], rotation=20)
    plt.ylabel("Score")
    plt.title("Benchmark - F1 and AUPRC by model")
    plt.legend()
    _save_figure("results/graphs/04_f1_auprc_benchmark.png")

    # Graph 5: Drift score distribution.
    y_true = np.asarray(abd["y_true"])
    drift_scores = np.asarray(abd["drift_scores"])

    plt.figure(figsize=(8, 6))
    plt.hist(drift_scores[y_true == 0], bins=60, alpha=0.6, color="green", label="Genuine")
    plt.hist(drift_scores[y_true == 1], bins=60, alpha=0.6, color="red", label="Fraud")
    plt.yscale("log")
    plt.axvline(abd["drift_threshold"], color="black", linestyle="--", label="Drift threshold")
    plt.title("Drift score distribution - genuine vs fraud")
    plt.xlabel("drift_score")
    plt.ylabel("Count")
    plt.legend()
    _save_figure("results/graphs/05_drift_score_distribution.png")

    # Graph 6: Graph anomaly score distribution.
    graph_scores = np.asarray(abd["graph_scores"])
    plt.figure(figsize=(8, 6))
    plt.hist(graph_scores[y_true == 0], bins=60, alpha=0.6, color="green", label="Genuine")
    plt.hist(graph_scores[y_true == 1], bins=60, alpha=0.6, color="red", label="Fraud")
    plt.yscale("log")
    plt.title("Graph anomaly score distribution - genuine vs fraud")
    plt.xlabel("graph_anomaly_score")
    plt.ylabel("Count")
    plt.legend()
    _save_figure("results/graphs/06_graph_score_distribution.png")

    # Graph 7: Fusion score density.
    fusion_scores = np.asarray(abd["fusion_scores"])
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=fusion_scores[y_true == 0], fill=True, alpha=0.3, color="green", label="Genuine")
    sns.kdeplot(x=fusion_scores[y_true == 1], fill=True, alpha=0.3, color="red", label="Fraud")
    plt.axvline(abd["optimal_threshold"], color="black", linestyle="--", label="Optimal threshold")
    plt.title("Fusion score density - genuine vs fraud")
    plt.xlabel("fusion_score")
    plt.ylabel("Density")
    plt.legend()
    _save_figure("results/graphs/07_fusion_score_density.png")

    # Graph 8: GNN train loss and val AUPRC.
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    epochs = gnn_history["epoch"]
    ax1.plot(epochs, gnn_history["train_loss"], color="#1D9E75", linewidth=2, label="Train loss")
    ax2.plot(epochs, gnn_history["val_auprc"], color="#E24B4A", linewidth=2, label="Val AUPRC")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax2.set_ylabel("Validation AUPRC")
    ax1.set_title("GNN training loss and validation AUPRC")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    _save_figure("results/graphs/08_gnn_training_loss.png")
