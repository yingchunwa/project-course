"""PCA-based motor fault detector.

The detector is intentionally lightweight for edge devices such as Orange Pi:
only NumPy, pandas, scikit-learn, and joblib are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAPrediction:
    scores: np.ndarray
    threshold: float
    predictions: np.ndarray


class PCAFaultDetector:
    """StandardScaler -> PCA reconstruction-error detector.

    The model must be trained with normal samples only. A sample whose
    reconstruction error is greater than the learned threshold is classified
    as fault; otherwise it is classified as normal.
    """

    def __init__(self, n_components: float = 0.99, threshold_quantile: float = 0.95):
        self.n_components = n_components
        self.threshold_quantile = threshold_quantile
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.threshold_: float | None = None
        self.feature_columns_: list[str] | None = None
        self.normal_train_scores_: np.ndarray | None = None

    def fit(self, normal_features: pd.DataFrame) -> "PCAFaultDetector":
        if normal_features.empty:
            raise ValueError("No normal samples found for PCA training.")

        self.feature_columns_ = list(normal_features.columns)
        x_scaled = self.scaler.fit_transform(normal_features.to_numpy(dtype=float))
        z = self.pca.fit_transform(x_scaled)
        x_reconstructed = self.pca.inverse_transform(z)
        scores = self._reconstruction_error(x_scaled, x_reconstructed)

        self.normal_train_scores_ = scores
        self.threshold_ = float(np.quantile(scores, self.threshold_quantile))
        return self

    def predict(self, features: pd.DataFrame) -> PCAPrediction:
        self._check_fitted()
        self._check_columns(features)

        x_scaled = self.scaler.transform(features.to_numpy(dtype=float))
        z = self.pca.transform(x_scaled)
        x_reconstructed = self.pca.inverse_transform(z)
        scores = self._reconstruction_error(x_scaled, x_reconstructed)
        predictions = np.where(scores > self.threshold_, "fault", "normal")

        return PCAPrediction(
            scores=scores,
            threshold=float(self.threshold_),
            predictions=predictions,
        )

    def save(self, model_path: str | Path) -> None:
        self._check_fitted()
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)

    @classmethod
    def load(cls, model_path: str | Path) -> "PCAFaultDetector":
        model = joblib.load(model_path)
        if not isinstance(model, cls):
            raise TypeError(f"Model file does not contain {cls.__name__}.")
        model._check_fitted()
        return model

    @staticmethod
    def _reconstruction_error(x_scaled: np.ndarray, x_reconstructed: np.ndarray) -> np.ndarray:
        return np.mean((x_scaled - x_reconstructed) ** 2, axis=1)

    def _check_fitted(self) -> None:
        if self.threshold_ is None or self.feature_columns_ is None:
            raise RuntimeError("PCAFaultDetector has not been fitted.")

    def _check_columns(self, features: pd.DataFrame) -> None:
        expected = self.feature_columns_ or []
        actual = list(features.columns)
        if actual != expected:
            missing = [col for col in expected if col not in actual]
            extra = [col for col in actual if col not in expected]
            raise ValueError(
                "Feature columns do not match the trained model. "
                f"Missing={missing}, extra={extra}"
            )
