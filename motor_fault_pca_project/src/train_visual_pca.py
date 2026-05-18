from __future__ import annotations

from pca_detector import PCAFaultDetector
from utils import normal_only, project_root, read_feature_csv, split_features_and_label


def main() -> None:
    root = project_root()
    csv_path = root / "features" / "visual_motion_features.csv"
    model_path = root / "models" / "visual_pca_model.pkl"

    df = read_feature_csv(csv_path)
    features, labels, _ = split_features_and_label(df)
    normal_features = normal_only(features, labels)

    detector = PCAFaultDetector(n_components=0.99, threshold_quantile=0.95)
    detector.fit(normal_features)
    detector.save(model_path)

    print(f"Saved visual PCA model: {model_path}")
    print(f"Features: {len(detector.feature_columns_)}")
    print(f"PCA components: {detector.pca.n_components_}")
    print(f"Threshold: {detector.threshold_}")


if __name__ == "__main__":
    main()
