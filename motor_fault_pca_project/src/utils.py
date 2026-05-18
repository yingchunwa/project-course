"""Shared utilities for PCA motor fault detection scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


META_COLUMNS = {
    "sample_id",
    "run_id",
    "window_id",
    "start_time",
    "end_time",
    "label",
    "rpm",
    "sensor_sample_rate_hz",
    "sensor_window_duration_s",
    "imu_sample_count",
    "cam_frame_count",
    "sync_offset_ms",
    "sync_drift_ppm",
    "seq_gap_count",
    "score",
    "threshold",
    "prediction",
    "model_path",
    "branch",
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "analysis_fps",
    "tracked_points",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_feature_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def select_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [col for col in df.columns if col not in META_COLUMNS]
    numeric_columns = [
        col for col in candidates if pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_columns:
        raise ValueError("No numeric feature columns found after excluding metadata columns.")
    return numeric_columns


def select_vibration_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select vibration PCA features that are less sensitive to gravity offset.

    Raw acceleration mean/rms/min/max/mean_abs are dominated by sensor mounting
    angle and static gravity. For realtime detection, use AC and spectral
    features so the model reacts to vibration-pattern changes instead of tiny
    DC offset drift.
    """
    columns = select_numeric_feature_columns(df)
    dc_sensitive_suffixes = (
        "_mean",
        "_rms",
        "_min",
        "_max",
        "_mean_abs",
    )
    return [
        col
        for col in columns
        if not (
            col.startswith("sensor_")
            and any(col.endswith(suffix) for suffix in dc_sensitive_suffixes)
        )
    ]


def split_features_and_label(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    if feature_columns is None:
        feature_columns = select_numeric_feature_columns(df)

    features = df.loc[:, feature_columns].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    if features.isna().any().any():
        bad_cols = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Feature data contains NaN or inf values in columns: {bad_cols}")

    labels = df["label"].astype(str).str.lower() if "label" in df.columns else None
    return features, labels, feature_columns


def normal_only(features: pd.DataFrame, labels: pd.Series | None) -> pd.DataFrame:
    if labels is None:
        raise ValueError("Training requires a label column with normal/fault values.")
    return features.loc[labels == "normal"].copy()


def build_result_frame(
    source_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    predictions: np.ndarray,
) -> pd.DataFrame:
    id_columns = [
        col
        for col in ["sample_id", "run_id", "window_id", "start_time", "end_time", "rpm", "label"]
        if col in source_df.columns
    ]
    result = source_df.loc[:, id_columns].copy()
    result["score"] = scores
    result["threshold"] = threshold
    result["prediction"] = predictions
    return result


def save_csv(df: pd.DataFrame, csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def compute_metrics(labels: pd.Series | None, predictions: np.ndarray, scores: np.ndarray) -> dict:
    if labels is None:
        return {}

    labels = labels.astype(str).str.lower()
    valid = labels.isin(["normal", "fault"])
    if not valid.all():
        invalid = sorted(labels.loc[~valid].unique().tolist())
        raise ValueError(f"Unsupported labels found: {invalid}. Expected normal/fault.")

    y_true = (labels == "fault").astype(int).to_numpy()
    y_pred = (predictions == "fault").astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = roc_auc_score(y_true, scores)
    else:
        metrics["auroc"] = None

    return metrics


def print_metrics(metrics: dict) -> None:
    if not metrics:
        print("No label column found. Metrics were skipped.")
        return

    print("Metrics:")
    for key in ["accuracy", "precision", "recall", "f1", "auroc"]:
        print(f"  {key}: {metrics[key]}")
    print("  confusion_matrix [[TN, FP], [FN, TP]]:")
    print(f"  {metrics['confusion_matrix']}")
