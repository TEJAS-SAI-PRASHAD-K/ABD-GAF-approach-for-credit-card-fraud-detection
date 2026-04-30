import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def _min_max_normalize(values: np.ndarray, bounds: tuple[float, float] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    if bounds is None:
        v_min = float(np.min(arr))
        v_max = float(np.max(arr))
    else:
        v_min, v_max = float(bounds[0]), float(bounds[1])

    denom = v_max - v_min
    if abs(denom) < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - v_min) / (denom + 1e-12)


def fuse_scores(
    drift_scores,
    graph_scores,
    w_drift: float = 0.4,
    w_graph: float = 0.6,
    drift_bounds: tuple[float, float] | None = None,
    graph_bounds: tuple[float, float] | None = None,
):
    drift_arr = np.asarray(drift_scores, dtype=float)
    graph_arr = np.asarray(graph_scores, dtype=float)

    if drift_arr.shape != graph_arr.shape:
        raise ValueError("drift_scores and graph_scores must have identical shapes.")

    norm_drift = _min_max_normalize(drift_arr, bounds=drift_bounds)
    norm_graph = _min_max_normalize(graph_arr, bounds=graph_bounds)

    fusion_score = (w_drift * norm_drift) + (w_graph * norm_graph)
    return fusion_score


def find_optimal_threshold(fusion_scores, y_val):
    best_threshold = 0.5
    best_f1 = -1.0

    y_val_arr = np.asarray(y_val, dtype=int)
    fusion_arr = np.asarray(fusion_scores, dtype=float)

    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (fusion_arr >= threshold).astype(int)
        f1 = f1_score(y_val_arr, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    print(f"Optimal fusion threshold selected from validation set: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


def classify(fusion_scores, threshold):
    fusion_arr = np.asarray(fusion_scores, dtype=float)
    return (fusion_arr >= threshold).astype(int)


def tune_fusion_weights(
    drift_val,
    graph_val,
    y_val,
    drift_bounds: tuple[float, float],
    graph_bounds: tuple[float, float],
):
    best = {
        "w_drift": 0.4,
        "w_graph": 0.6,
        "auprc": -1.0,
        "f1": -1.0,
    }

    y_val_arr = np.asarray(y_val, dtype=int)

    for w_drift in np.arange(0.0, 1.01, 0.1):
        w_drift = round(float(w_drift), 1)
        w_graph = round(1.0 - w_drift, 1)

        fused = fuse_scores(
            drift_scores=drift_val,
            graph_scores=graph_val,
            w_drift=w_drift,
            w_graph=w_graph,
            drift_bounds=drift_bounds,
            graph_bounds=graph_bounds,
        )
        auprc = average_precision_score(y_val_arr, fused)

        threshold = find_optimal_threshold(fused, y_val_arr)
        preds = classify(fused, threshold)
        f1 = f1_score(y_val_arr, preds, zero_division=0)

        if (auprc > best["auprc"]) or (np.isclose(auprc, best["auprc"]) and f1 > best["f1"]):
            best = {
                "w_drift": w_drift,
                "w_graph": w_graph,
                "auprc": float(auprc),
                "f1": float(f1),
                "threshold": float(threshold),
            }

    print(
        "Best fusion weights from validation tuning: "
        f"w_drift={best['w_drift']:.2f}, w_graph={best['w_graph']:.2f}, "
        f"AUPRC={best['auprc']:.4f}, F1={best['f1']:.4f}"
    )
    return best
