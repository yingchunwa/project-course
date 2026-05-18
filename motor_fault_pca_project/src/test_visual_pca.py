from __future__ import annotations

from pca_detector import PCAFaultDetector
from utils import (
    build_result_frame,
    compute_metrics,
    print_metrics,
    project_root,
    read_feature_csv,
    save_csv,
    split_features_and_label,
)


def main() -> None:
    root = project_root()
    csv_path = root / "features" / "visual_motion_features.csv"
    model_path = root / "models" / "visual_pca_model.pkl"
    result_path = root / "results" / "visual_pca_results.csv"

    detector = PCAFaultDetector.load(model_path)
    df = read_feature_csv(csv_path)
    features, labels, _ = split_features_and_label(df, detector.feature_columns_)

    output = detector.predict(features)
    result = build_result_frame(df, output.scores, output.threshold, output.predictions)
    save_csv(result, result_path)

    print(f"Saved visual PCA results: {result_path}")
    print_metrics(compute_metrics(labels, output.predictions, output.scores))


if __name__ == "__main__":
    main()
