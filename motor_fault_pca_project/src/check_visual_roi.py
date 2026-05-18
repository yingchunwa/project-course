from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from realtime_features import _validate_roi
from utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a camera preview with ROI and tracked corners.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=140)
    parser.add_argument("--fps", type=int, default=420)
    parser.add_argument("--fourcc", default="YUY2")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--max-corners", type=int, default=80)
    parser.add_argument("--output", default="results/visual_roi_check.jpg")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read a camera frame.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    height, width = gray.shape
    roi = tuple(args.roi) if args.roi else (0, 0, width, height)
    x, y, w, h = _validate_roi(width, height, roi)

    mask = np.zeros_like(gray_eq)
    mask[y : y + h, x : x + w] = 255
    corners = cv2.goodFeaturesToTrack(
        gray_eq,
        maxCorners=args.max_corners,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=7,
        mask=mask,
    )

    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 1)
    corner_count = 0
    if corners is not None:
        for point in corners.reshape(-1, 2):
            px, py = int(round(point[0])), int(round(point[1]))
            cv2.circle(preview, (px, py), 2, (0, 0, 255), -1)
            corner_count += 1

    output_path = project_root() / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
    print(f"saved: {output_path}")
    print(f"roi: x={x}, y={y}, w={w}, h={h}")
    print(f"corners: {corner_count}")


if __name__ == "__main__":
    main()
