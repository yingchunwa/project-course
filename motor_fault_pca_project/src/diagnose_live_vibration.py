from __future__ import annotations

import argparse
import numpy as np

from pca_detector import PCAFaultDetector
from realtime_features import align_feature_dict, vibration_window_from_i2c
from utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose one live vibration PCA window.")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate-hz", type=int, default=60)
    parser.add_argument("--i2c-bus", type=int, default=7)
    parser.add_argument("--i2c-addr", type=lambda value: int(value, 0), default=0x6A)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    root = project_root()
    detector = PCAFaultDetector.load(root / "models" / "vibration_pca_model.pkl")
    record = vibration_window_from_i2c(
        bus_id=args.i2c_bus,
        address=args.i2c_addr,
        window_seconds=args.window_seconds,
        sample_rate_hz=args.sample_rate_hz,
    )
    features = align_feature_dict(record.features, detector.feature_columns_)

    x_scaled = detector.scaler.transform(features.to_numpy(dtype=float))
    z = detector.pca.transform(x_scaled)
    x_reconstructed = detector.pca.inverse_transform(z)
    residual = (x_scaled[0] - x_reconstructed[0]) ** 2
    z_abs = np.abs(x_scaled[0])
    output = detector.predict(features)

    print(f"score: {output.scores[0]}")
    print(f"threshold: {output.threshold}")
    print(f"prediction: {output.predictions[0]}")
    print("Top reconstruction residual features:")
    for index in np.argsort(residual)[::-1][: args.top]:
        name = detector.feature_columns_[index]
        print(f"  {name}: residual={residual[index]:.6f}, z={z_abs[index]:.3f}, value={features.iloc[0, index]}")


if __name__ == "__main__":
    main()
