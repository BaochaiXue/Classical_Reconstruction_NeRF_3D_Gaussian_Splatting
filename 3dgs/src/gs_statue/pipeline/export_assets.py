"""Wrapper utilities for ns-export commands."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..utils import ensure_executable, run_command


def _run_export(
    subcommand: str,
    config_path: Path,
    output_dir: Path,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    ensure_executable("ns-export")
    if not config_path.exists():
        raise FileNotFoundError(f"Nerfstudio config not found: {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ns-export",
        subcommand,
        "--load-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if extra_args:
        cmd += list(extra_args)

    run_command(cmd)


def export_gaussian_splat(
    config_path: Path,
    output_dir: Path,
    *,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Export Gaussian splats as .ply files."""
    _run_export("gaussian-splat", config_path, output_dir, extra_args)


def export_cameras(
    config_path: Path,
    output_dir: Path,
    *,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Export camera poses for diagnostics/plots."""
    _run_export("cameras", config_path, output_dir, extra_args)


def export_pointcloud(
    config_path: Path,
    output_dir: Path,
    *,
    num_points: int | None = None,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Export a point cloud for baseline comparisons."""
    args = list(extra_args or [])
    if num_points is not None:
        args += ["--num-points", str(num_points)]
    _run_export("pointcloud", config_path, output_dir, args)
