from __future__ import annotations

import numpy as np

from pca_detector import PCAFaultDetector
from utils import (
    project_root,
    read_feature_csv,
    select_vibration_feature_columns,
    split_features_and_label,
)


def main() -> None:
    root = project_root()
    csv_path = root / "features" / "vibration_features.csv"
    model_path = root / "models" / "vibration_pca_model.pkl"

    df = read_feature_csv(csv_path)
    feature_columns = select_vibration_feature_columns(df)
    features, labels, feature_columns = split_features_and_label(df, feature_columns)
    normal_count = int((labels == "normal").sum()) if labels is not None else 0
    fault_count = int((labels == "fault").sum()) if labels is not None else 0

    print(f"CSV rows: {len(df)}")
    print(f"normal rows: {normal_count}")
    print(f"fault rows: {fault_count}")
    print(f"PCA feature columns: {len(feature_columns)}")
    print("First 10 feature columns:")
    for name in feature_columns[:10]:
        print(f"  {name}")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    detector = PCAFaultDetector.load(model_path)
    print(f"Model threshold: {detector.threshold_}")
    print(f"Model PCA components: {detector.pca.n_components_}")
    if detector.normal_train_scores_ is not None:
        scores = detector.normal_train_scores_
        print(f"Train score min: {float(np.min(scores))}")
        print(f"Train score median: {float(np.median(scores))}")
        print(f"Train score p95: {float(np.quantile(scores, 0.95))}")
        print(f"Train score max: {float(np.max(scores))}")

    output = detector.predict(features.loc[:, detector.feature_columns_])
    print(f"CSV score median: {float(np.median(output.scores))}")
    print(f"CSV score p95: {float(np.quantile(output.scores, 0.95))}")
    print(f"CSV score max: {float(np.max(output.scores))}")


if __name__ == "__main__":
    main()
