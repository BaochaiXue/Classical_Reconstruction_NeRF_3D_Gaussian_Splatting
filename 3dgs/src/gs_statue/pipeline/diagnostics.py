"""Diagnostic plots for the statue pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


def _iter_camera_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob("*.json"))
    else:
        raise FileNotFoundError(f"Camera path not found: {path}")


def _extract_centers(payload: Union[dict, Sequence[dict]]) -> List[np.ndarray]:
    if isinstance(payload, list):
        cameras = payload
    else:
        cameras = (
            payload.get("cameras")
            or payload.get("camera_path")
            or payload.get("frames")
            or []
        )
    centers: List[np.ndarray] = []
    for cam in cameras:
        matrix = (
            cam.get("camera_to_world")
            or cam.get("c2w")
            or cam.get("camera_to_world_matrix")
            or cam.get("transform")
            or cam.get("transform_matrix")
        )
        if matrix is None:
            continue
        arr = np.asarray(matrix, dtype=float)
        if arr.size == 16:
            arr = arr.reshape(4, 4)
        elif arr.shape == (3, 4):
            arr = np.vstack([arr, np.array([0.0, 0.0, 0.0, 1.0])])
        elif arr.shape == (4, 4):
            pass
        else:
            continue
        centers.append(arr[:3, 3])
    return centers


def load_camera_centers(path: Path) -> np.ndarray:
    """Load all camera centers from JSON files."""
    all_centers: List[np.ndarray] = []
    for json_file in _iter_camera_files(path):
        data = json.loads(json_file.read_text())
        all_centers.extend(_extract_centers(data))
    if not all_centers:
        raise RuntimeError(f"No cameras found in {path}")
    return np.stack(all_centers, axis=0)


def plot_camera_poses(centers: np.ndarray, output_path: Path) -> None:
    """Plot camera trajectories in 3D."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=8, c="tab:blue")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera trajectory")
    ax.view_init(30, 30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot camera poses exported by ns-export cameras."
    )
    parser.add_argument("cameras_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    centers = load_camera_centers(args.cameras_path)
    plot_camera_poses(centers, args.output_path)


if __name__ == "__main__":
    main()
