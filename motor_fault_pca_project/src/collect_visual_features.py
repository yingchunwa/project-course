from __future__ import annotations

import argparse
import time

from realtime_features import (
    append_feature_row,
    visual_motion_window_from_camera,
    visual_vibration_window_from_camera,
)
from utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect visual motion PCA training features.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=140)
    parser.add_argument("--fps", type=int, default=420)
    parser.add_argument("--fourcc", default="YUY2")
    parser.add_argument("--method", choices=["lk", "motion"], default="lk")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--max-corners", type=int, default=80)
    parser.add_argument("--min-frequency", type=float, default=1.0)
    parser.add_argument("--max-frequency", type=float)
    parser.add_argument("--label", choices=["normal", "fault"], required=True)
    parser.add_argument("--windows", type=int, default=30)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--output", default="features/visual_motion_features.csv")
    args = parser.parse_args()

    root = project_root()
    output = root / args.output

    for window_id in range(args.windows):
        if args.method == "lk":
            record = visual_vibration_window_from_camera(
                camera_index=args.camera_index,
                window_seconds=args.window_seconds,
                width=args.width,
                height=args.height,
                fps=args.fps,
                fourcc=args.fourcc,
                roi=tuple(args.roi) if args.roi else None,
                max_corners=args.max_corners,
                min_frequency=args.min_frequency,
                max_frequency=args.max_frequency,
            )
        else:
            record = visual_motion_window_from_camera(
                camera_index=args.camera_index,
                window_seconds=args.window_seconds,
                width=args.width,
                height=args.height,
                fps=args.fps,
                fourcc=args.fourcc,
            )
        row = {
            "sample_id": f"visual_{int(time.time())}_{window_id}",
            "run_id": "visual_runtime_collect",
            "window_id": window_id,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "label": args.label,
            **record.features,
        }
        append_feature_row(output, row)
        print(f"Saved visual window {window_id + 1}/{args.windows}: {args.label}")


if __name__ == "__main__":
    main()
