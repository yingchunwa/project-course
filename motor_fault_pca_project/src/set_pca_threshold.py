from __future__ import annotations

import argparse

from pca_detector import PCAFaultDetector
from utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Set a fixed PCA detector threshold.")
    parser.add_argument("--branch", choices=["visual", "vibration"], required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    root = project_root()
    model_path = root / "models" / f"{args.branch}_pca_model.pkl"
    detector = PCAFaultDetector.load(model_path)
    old_threshold = detector.threshold_
    detector.threshold_ = float(args.threshold)
    detector.save(model_path)

    print(f"Updated {args.branch} threshold")
    print(f"  model: {model_path}")
    print(f"  old_threshold: {old_threshold}")
    print(f"  new_threshold: {detector.threshold_}")


if __name__ == "__main__":
    main()
