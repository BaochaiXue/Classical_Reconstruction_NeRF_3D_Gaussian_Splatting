"""Rendering helpers using ns-render."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..utils import ensure_executable, run_command


def render_dataset_views(
    config_path: Path,
    output_dir: Path,
    *,
    split: str = "train+test",
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Render train/test views for qualitative checks."""
    ensure_executable("ns-render")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-render",
        "dataset",
        "--load-config",
        str(config_path),
        "--output-path",
        str(output_dir),
        "--rendered-output-names",
        "rgb",
        "--skip-eval",
        "--split",
        split,
    ]
    if extra_args:
        cmd += list(extra_args)
    run_command(cmd)


def render_orbit_video(
    config_path: Path,
    camera_path: Path,
    output_path: Path,
    *,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Render a camera-path video (e.g., orbit)."""
    ensure_executable("ns-render")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-render",
        "camera-path",
        "--load-config",
        str(config_path),
        "--camera-path",
        str(camera_path),
        "--output-path",
        str(output_path),
    ]
    if extra_args:
        cmd += list(extra_args)
    run_command(cmd)


def render_interpolate_video(
    config_path: Path,
    output_path: Path,
    *,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Render an interpolation video without a camera path."""
    ensure_executable("ns-render")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-render",
        "interpolate",
        "--load-config",
        str(config_path),
        "--output-path",
        str(output_path),
    ]
    if extra_args:
        cmd += list(extra_args)
    run_command(cmd)
