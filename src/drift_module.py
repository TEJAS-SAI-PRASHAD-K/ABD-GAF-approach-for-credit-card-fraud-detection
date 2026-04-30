from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftBaseline:
    mean_amount: float
    std_amount: float
    mean_time_gap: float
    std_time_gap: float


class BehavioralDriftDetector:
    def __init__(self, window: int = 50, epsilon: float = 1e-8):
        self.window = window
        self.epsilon = epsilon
        self.baseline: DriftBaseline | None = None
        self.drift_threshold: float | None = None

    def fit(self, X_train: pd.DataFrame) -> None:
        required = {"Time", "Amount"}
        if not required.issubset(X_train.columns):
            raise ValueError("X_train must include 'Time' and 'Amount' columns.")

        amount_series = X_train["Amount"].astype(float)
        time_gap = X_train["Time"].astype(float).diff().fillna(0.0)

        rolling_mean_amount = amount_series.rolling(window=self.window, min_periods=1).mean()
        rolling_std_amount = amount_series.rolling(window=self.window, min_periods=1).std().fillna(0.0)

        self.baseline = DriftBaseline(
            mean_amount=float(rolling_mean_amount.mean()),
            std_amount=float(max(rolling_std_amount.mean(), self.epsilon)),
            mean_time_gap=float(time_gap.mean()),
            std_time_gap=float(max(time_gap.std(), self.epsilon)),
        )

        train_scores = self.compute_drift_score(X_train)
        self.drift_threshold = float(np.percentile(train_scores, 97.5))

    def compute_drift_score(self, X: pd.DataFrame) -> np.ndarray:
        if self.baseline is None:
            raise RuntimeError("BehavioralDriftDetector.fit must be called before scoring.")
        if not {"Time", "Amount"}.issubset(X.columns):
            raise ValueError("X must include 'Time' and 'Amount' columns.")

        amount_series = X["Amount"].astype(float)
        time_gap = X["Time"].astype(float).diff().fillna(0.0)

        rolling_mean_amount = amount_series.rolling(window=self.window, min_periods=1).mean()
        rolling_std_amount = amount_series.rolling(window=self.window, min_periods=1).std().fillna(0.0)

        z_amount = (amount_series - rolling_mean_amount) / (rolling_std_amount + self.epsilon)
        z_time = (time_gap - self.baseline.mean_time_gap) / (self.baseline.std_time_gap + self.epsilon)

        drift_score = np.sqrt(np.square(z_amount.to_numpy()) + np.square(z_time.to_numpy()))
        return drift_score

    def flag_anomalies(self, drift_scores: np.ndarray) -> np.ndarray:
        if self.drift_threshold is None:
            raise RuntimeError("Drift threshold is not available. Call fit first.")
        return (drift_scores > self.drift_threshold).astype(int)
