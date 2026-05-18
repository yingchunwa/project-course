from __future__ import annotations

import argparse

import numpy as np

from pca_detector import PCAFaultDetector
from realtime_features import align_feature_dict, vibration_window_from_i2c
from utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live vibration scores without saving.")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate-hz", type=int, default=60)
    parser.add_argument("--i2c-bus", type=int, default=7)
    parser.add_argument("--i2c-addr", type=lambda value: int(value, 0), default=0x6A)
    args = parser.parse_args()

    root = project_root()
    detector = PCAFaultDetector.load(root / "models" / "vibration_pca_model.pkl")
    scores = []

    for index in range(args.windows):
        record = vibration_window_from_i2c(
            bus_id=args.i2c_bus,
            address=args.i2c_addr,
            window_seconds=args.window_seconds,
            sample_rate_hz=args.sample_rate_hz,
        )
        features = align_feature_dict(record.features, detector.feature_columns_)
        output = detector.predict(features)
        score = float(output.scores[0])
        scores.append(score)
        print(
            f"{index + 1}/{args.windows}: "
            f"score={score:.6f}, threshold={output.threshold:.6f}, "
            f"prediction={output.predictions[0]}"
        )

    scores_array = np.asarray(scores, dtype=float)
    print("Summary:")
    print(f"  min={float(np.min(scores_array))}")
    print(f"  median={float(np.median(scores_array))}")
    print(f"  p95={float(np.quantile(scores_array, 0.95))}")
    print(f"  max={float(np.max(scores_array))}")


if __name__ == "__main__":
    main()
