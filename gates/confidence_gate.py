"""ML confidence gate for SignalFusion V2.

Predicts P(signal_correct) using logistic regression trained on
historical signal features. Gates out signals below a calibrated threshold.
"""

import pickle
from pathlib import Path

import numpy as np


class ConfidenceGate:
    """Logistic regression gate that predicts signal correctness."""

    FEATURES = [
        "composite_score",
        "n_sources",
        "spread_cents",
        "hours_to_close_log",
        "book_imbalance",
        "yes_mid",
        "source_ph_int",
    ]

    def __init__(self, model_path: str | None = None):
        self.model = None       # LogisticRegression
        self.scaler = None      # StandardScaler
        self.threshold = 0.55   # conservative default
        self.enabled = False

        if model_path:
            self.load(model_path)

    def _prepare_features(self, features: dict) -> np.ndarray:
        """Extract and transform the feature vector from a signal dict."""
        hours = max(features.get("hours_to_close", 1.0), 0.01)
        row = [
            features.get("composite_score", 0.0),
            features.get("n_sources", 1),
            features.get("spread_cents", 5),
            np.log1p(hours),
            features.get("book_imbalance", 0.0),
            features.get("yes_mid", 50),
            1.0 if features.get("source_ph") else 0.0,
        ]
        return np.array(row).reshape(1, -1)

    def predict_confidence(self, features: dict) -> float:
        """Returns P(signal_correct) from the trained model."""
        if not self.enabled or self.model is None:
            return 0.5  # neutral if no model

        X = self._prepare_features(features)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        prob = self.model.predict_proba(X)[0, 1]
        return float(prob)

    def should_pass(self, features: dict) -> bool:
        """Returns True if the model's confidence meets the threshold."""
        if not self.enabled:
            return True  # pass everything if ML gate disabled
        return self.predict_confidence(features) >= self.threshold

    def save(self, path: str) -> None:
        """Save model, scaler, and threshold to pickle."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "feature_names": self.FEATURES,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load model from pickle. Enables the gate if model exists."""
        p = Path(path)
        if not p.exists():
            self.enabled = False
            return
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data.get("scaler")
        self.threshold = data.get("threshold", 0.55)
        self.enabled = True
