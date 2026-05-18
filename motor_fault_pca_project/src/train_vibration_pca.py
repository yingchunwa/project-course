from __future__ import annotations

import argparse

from pca_detector import PCAFaultDetector
from utils import (
    normal_only,
    project_root,
    read_feature_csv,
    select_vibration_feature_columns,
    split_features_and_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vibration PCA fault detector.")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    args = parser.parse_args()

    root = project_root()
    csv_path = root / "features" / "vibration_features.csv"
    model_path = root / "models" / "vibration_pca_model.pkl"

    df = read_feature_csv(csv_path)
    feature_columns = select_vibration_feature_columns(df)
    features, labels, _ = split_features_and_label(df, feature_columns)
    normal_features = normal_only(features, labels)

    detector = PCAFaultDetector(
        n_components=0.99,
        threshold_quantile=args.threshold_quantile,
    )
    detector.fit(normal_features)
    detector.threshold_ = float(detector.threshold_ * args.threshold_scale)
    detector.save(model_path)

    print(f"Saved vibration PCA model: {model_path}")
    print(f"Features: {len(detector.feature_columns_)}")
    print(f"PCA components: {detector.pca.n_components_}")
    print(f"Threshold: {detector.threshold_}")


if __name__ == "__main__":
    main()
