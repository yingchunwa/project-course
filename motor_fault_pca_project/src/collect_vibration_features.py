from __future__ import annotations

import argparse
import time

from realtime_features import append_feature_row, vibration_window_from_serial
from realtime_features import vibration_window_from_i2c
from utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vibration PCA training features.")
    parser.add_argument("--source", choices=["i2c", "serial"], default="i2c")
    parser.add_argument("--port", help="Serial port, for example COM3 or /dev/ttyUSB0.")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--i2c-bus", type=int, default=7)
    parser.add_argument("--i2c-addr", type=lambda value: int(value, 0), default=0x6A)
    parser.add_argument("--sample-rate-hz", type=int, default=60)
    parser.add_argument("--include-gyro", action="store_true")
    parser.add_argument("--label", choices=["normal", "fault"], required=True)
    parser.add_argument("--windows", type=int, default=30)
    parser.add_argument("--window-seconds", type=float, default=0.25)
    parser.add_argument("--min-values-per-line", type=int, default=3)
    parser.add_argument(
        "--axis-start-index",
        type=int,
        default=0,
        help=(
            "Index where ax,ay,az begin in each serial line. "
            "Use 0 for 'ax,ay,az'; use 2 for 'seq,tick,ax,ay,az'."
        ),
    )
    parser.add_argument("--output", default="features/vibration_features.csv")
    args = parser.parse_args()

    root = project_root()
    output = root / args.output

    for window_id in range(args.windows):
        if args.source == "i2c":
            record = vibration_window_from_i2c(
                bus_id=args.i2c_bus,
                address=args.i2c_addr,
                window_seconds=args.window_seconds,
                sample_rate_hz=args.sample_rate_hz,
                include_gyro=args.include_gyro,
            )
        else:
            if not args.port:
                raise ValueError("--port is required when --source serial is used.")
            record = vibration_window_from_serial(
                port=args.port,
                baudrate=args.baudrate,
                window_seconds=args.window_seconds,
                sample_rate_hz=args.sample_rate_hz,
                min_values_per_line=args.min_values_per_line,
                axis_start_index=args.axis_start_index,
            )
        row = {
            "sample_id": f"vibration_{int(time.time())}_{window_id}",
            "run_id": "vibration_runtime_collect",
            "window_id": window_id,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "label": args.label,
            **record.features,
        }
        append_feature_row(output, row)
        print(f"Saved vibration window {window_id + 1}/{args.windows}: {args.label}")


if __name__ == "__main__":
    main()
