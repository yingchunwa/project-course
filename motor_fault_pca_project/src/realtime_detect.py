from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from pca_detector import PCAFaultDetector
from realtime_features import (
    align_feature_dict,
    append_feature_row,
    visual_motion_window_from_camera,
    visual_vibration_window_from_camera,
    vibration_window_from_i2c,
    vibration_window_from_serial,
)
from utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent realtime PCA fault detection.")
    parser.add_argument("--visual", action="store_true", help="Enable camera visual PCA branch.")
    parser.add_argument("--vibration", action="store_true", help="Enable vibration serial PCA branch.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=140)
    parser.add_argument("--fps", type=int, default=420)
    parser.add_argument("--fourcc", default="YUY2")
    parser.add_argument("--visual-method", choices=["lk", "motion"], default="lk")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--max-corners", type=int, default=80)
    parser.add_argument("--min-frequency", type=float, default=1.0)
    parser.add_argument("--max-frequency", type=float)
    parser.add_argument("--vibration-source", choices=["i2c", "serial"], default="i2c")
    parser.add_argument("--port", help="Serial port for vibration module, for example COM3 or /dev/ttyUSB0.")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--i2c-bus", type=int, default=7)
    parser.add_argument("--i2c-addr", type=lambda value: int(value, 0), default=0x6A)
    parser.add_argument("--sample-rate-hz", type=int, default=60)
    parser.add_argument("--include-gyro", action="store_true")
    parser.add_argument(
        "--axis-start-index",
        type=int,
        default=0,
        help=(
            "Index where ADXL345 ax,ay,az begin in each serial line. "
            "Use 0 for 'ax,ay,az'; use 2 for 'seq,tick,ax,ay,az'."
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=0.25)
    parser.add_argument("--interval-seconds", type=float, default=0.05)
    parser.add_argument(
        "--save-normal-candidates",
        action="store_true",
        help="Save high-confidence normal windows for periodic PCA retraining.",
    )
    parser.add_argument(
        "--normal-candidate-ratio",
        type=float,
        default=0.5,
        help="Save as normal candidate when score <= threshold * this ratio.",
    )
    parser.add_argument(
        "--candidate-dir",
        default="features/normal_candidates",
        help="Directory for high-confidence normal candidate CSV files.",
    )
    args = parser.parse_args()

    if not args.visual and not args.vibration:
        raise ValueError("Enable at least one branch: --visual or --vibration.")
    if args.vibration and args.vibration_source == "serial" and not args.port:
        raise ValueError("Vibration branch requires --port.")

    root = project_root()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    if args.visual:
        threads.append(
            threading.Thread(
                target=_visual_loop,
                args=(
                    root,
                    args.camera_index,
                    args.width,
                    args.height,
                    args.fps,
                    args.fourcc,
                    args.visual_method,
                    tuple(args.roi) if args.roi else None,
                    args.max_corners,
                    args.min_frequency,
                    args.max_frequency,
                    args.window_seconds,
                    args.interval_seconds,
                    args.save_normal_candidates,
                    args.normal_candidate_ratio,
                    args.candidate_dir,
                    stop_event,
                ),
                daemon=True,
            )
        )
    if args.vibration:
        threads.append(
            threading.Thread(
                target=_vibration_loop,
                args=(
                    root,
                    args.port,
                    args.baudrate,
                    args.vibration_source,
                    args.i2c_bus,
                    args.i2c_addr,
                    args.sample_rate_hz,
                    args.include_gyro,
                    args.axis_start_index,
                    args.window_seconds,
                    args.interval_seconds,
                    args.save_normal_candidates,
                    args.normal_candidate_ratio,
                    args.candidate_dir,
                    stop_event,
                ),
                daemon=True,
            )
        )

    for thread in threads:
        thread.start()

    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping realtime detection...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=3)


def _visual_loop(
    root: Path,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    fourcc: str,
    visual_method: str,
    roi: tuple[int, int, int, int] | None,
    max_corners: int,
    min_frequency: float,
    max_frequency: float | None,
    window_seconds: float,
    interval_seconds: float,
    save_normal_candidates: bool,
    normal_candidate_ratio: float,
    candidate_dir: str,
    stop_event: threading.Event,
) -> None:
    detector = PCAFaultDetector.load(root / "models" / "visual_pca_model.pkl")
    window_id = 0
    while not stop_event.is_set():
        if visual_method == "lk":
            record = visual_vibration_window_from_camera(
                camera_index=camera_index,
                window_seconds=window_seconds,
                width=width,
                height=height,
                fps=fps,
                fourcc=fourcc,
                roi=roi,
                max_corners=max_corners,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
            )
        else:
            record = visual_motion_window_from_camera(
                camera_index=camera_index,
                window_seconds=window_seconds,
                width=width,
                height=height,
                fps=fps,
                fourcc=fourcc,
            )
        row = align_feature_dict(record.features, detector.feature_columns_)
        output = detector.predict(row)
        _print_result("visual", output.scores[0], output.threshold, output.predictions[0])
        if save_normal_candidates:
            _save_candidate_if_confident(
                root=root,
                branch="visual",
                record_features=record.features,
                feature_columns=detector.feature_columns_,
                start_time=record.start_time,
                end_time=record.end_time,
                window_id=window_id,
                score=float(output.scores[0]),
                threshold=output.threshold,
                prediction=str(output.predictions[0]),
                normal_candidate_ratio=normal_candidate_ratio,
                candidate_dir=candidate_dir,
            )
        window_id += 1
        _sleep_or_stop(interval_seconds, stop_event)


def _vibration_loop(
    root: Path,
    port: str | None,
    baudrate: int,
    vibration_source: str,
    i2c_bus: int,
    i2c_addr: int,
    sample_rate_hz: int,
    include_gyro: bool,
    axis_start_index: int,
    window_seconds: float,
    interval_seconds: float,
    save_normal_candidates: bool,
    normal_candidate_ratio: float,
    candidate_dir: str,
    stop_event: threading.Event,
) -> None:
    detector = PCAFaultDetector.load(root / "models" / "vibration_pca_model.pkl")
    window_id = 0
    while not stop_event.is_set():
        if vibration_source == "i2c":
            record = vibration_window_from_i2c(
                bus_id=i2c_bus,
                address=i2c_addr,
                window_seconds=window_seconds,
                sample_rate_hz=sample_rate_hz,
                include_gyro=include_gyro,
            )
        else:
            record = vibration_window_from_serial(
                port=port or "",
                baudrate=baudrate,
                window_seconds=window_seconds,
                sample_rate_hz=sample_rate_hz,
                min_values_per_line=3,
                axis_start_index=axis_start_index,
            )
        row = align_feature_dict(record.features, detector.feature_columns_)
        output = detector.predict(row)
        _print_result("vibration", output.scores[0], output.threshold, output.predictions[0])
        if save_normal_candidates:
            _save_candidate_if_confident(
                root=root,
                branch="vibration",
                record_features=record.features,
                feature_columns=detector.feature_columns_,
                start_time=record.start_time,
                end_time=record.end_time,
                window_id=window_id,
                score=float(output.scores[0]),
                threshold=output.threshold,
                prediction=str(output.predictions[0]),
                normal_candidate_ratio=normal_candidate_ratio,
                candidate_dir=candidate_dir,
            )
        window_id += 1
        _sleep_or_stop(interval_seconds, stop_event)


def _print_result(branch: str, score: float, threshold: float, prediction: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{now}] {branch}: "
        f"score={score:.6f}, threshold={threshold:.6f}, prediction={prediction}"
    )


def _sleep_or_stop(seconds: float, stop_event: threading.Event) -> None:
    if seconds > 0:
        stop_event.wait(seconds)


def _save_candidate_if_confident(
    root: Path,
    branch: str,
    record_features: dict[str, float],
    feature_columns: list[str],
    start_time: float,
    end_time: float,
    window_id: int,
    score: float,
    threshold: float,
    prediction: str,
    normal_candidate_ratio: float,
    candidate_dir: str,
) -> None:
    if prediction != "normal" or score > threshold * normal_candidate_ratio:
        return

    now = int(time.time())
    row = {
        "sample_id": f"{branch}_normal_candidate_{now}_{window_id}",
        "run_id": f"{branch}_realtime_candidates",
        "window_id": window_id,
        "start_time": start_time,
        "end_time": end_time,
        "label": "normal",
        "score": score,
        "threshold": threshold,
        "prediction": prediction,
    }
    row.update({name: record_features[name] for name in feature_columns})
    append_feature_row(root / candidate_dir / f"{branch}_normal_candidates.csv", row)


if __name__ == "__main__":
    main()
