from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from utils import project_root


PIPELINE = [
    ("train visual PCA", "train_visual_pca.py"),
    ("test visual PCA", "test_visual_pca.py"),
    ("train vibration PCA", "train_vibration_pca.py"),
    ("test vibration PCA", "test_vibration_pca.py"),
]


def main() -> None:
    root = project_root()
    src_dir = root / "src"
    _check_inputs(root)

    for step_name, script_name in PIPELINE:
        print(f"\n=== {step_name} ===")
        subprocess.run(
            [sys.executable, str(src_dir / script_name)],
            cwd=root,
            check=True,
        )


def _check_inputs(root: Path) -> None:
    required_files = [
        root / "features" / "visual_motion_features.csv",
        root / "features" / "vibration_features.csv",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing feature CSV files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


if __name__ == "__main__":
    main()
