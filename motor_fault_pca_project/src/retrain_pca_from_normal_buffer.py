from __future__ import annotations

import argparse
import shutil
from datetime import datetime

from pca_detector import PCAFaultDetector
from utils import (
    normal_only,
    project_root,
    read_feature_csv,
    select_vibration_feature_columns,
    split_features_and_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Periodically retrain one PCA branch from high-confidence normal windows."
    )
    parser.add_argument("--branch", choices=["visual", "vibration"], required=True)
    parser.add_argument("--min-windows", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--threshold-scale", type=float, default=1.2)
    parser.add_argument(
        "--candidate-dir",
        default="features/normal_candidates",
        help="Directory containing branch normal candidate CSV files.",
    )
    args = parser.parse_args()

    root = project_root()
    candidate_path = root / args.candidate_dir / f"{args.branch}_normal_candidates.csv"
    model_path = root / "models" / f"{args.branch}_pca_model.pkl"
    backup_dir = root / "models" / "backups"

    df = read_feature_csv(candidate_path)
    if len(df) < args.min_windows:
        raise ValueError(
            f"Not enough normal candidates for retraining: {len(df)} < {args.min_windows}."
        )

    df = df.tail(args.max_windows).copy()
    if args.branch == "vibration":
        feature_columns = select_vibration_feature_columns(df)
    else:
        feature_columns = None

    features, labels, _ = split_features_and_label(df, feature_columns)
    normal_features = normal_only(features, labels)
    if len(normal_features) < args.min_windows:
        raise ValueError(
            f"Not enough labeled normal rows after filtering: {len(normal_features)} < {args.min_windows}."
        )

    detector = PCAFaultDetector(
        n_components=0.99,
        threshold_quantile=args.threshold_quantile,
    )
    detector.fit(normal_features)
    detector.threshold_ = float(detector.threshold_ * args.threshold_scale)

    if model_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"{args.branch}_pca_model_{stamp}.pkl"
        shutil.copy2(model_path, backup_path)
        print(f"Backed up old model: {backup_path}")

    detector.save(model_path)
    print(f"Saved retrained {args.branch} model: {model_path}")
    print(f"Training windows: {len(normal_features)}")
    print(f"Features: {len(detector.feature_columns_)}")
    print(f"PCA components: {detector.pca.n_components_}")
    print(f"Threshold: {detector.threshold_}")


if __name__ == "__main__":
    main()
