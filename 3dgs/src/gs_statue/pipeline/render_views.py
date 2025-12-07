"""Rendering helpers built on ns-render."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..utils import ensure_executable, run_command


def render_dataset_views(
    config_path: Path,
    output_path: Path,
    *,
    split: str = "train+test",
    rendered_outputs: Sequence[str] | None = None,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Re-render training/test images for qualitative comparisons."""
    ensure_executable("ns-render")
    if not config_path.exists():
        raise FileNotFoundError(f"Nerfstudio config not found: {config_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ns-render",
        "dataset",
        "--load-config",
        str(config_path),
        "--output-path",
        str(output_path),
        "--split",
        split,
    ]

    outputs = rendered_outputs or ("rgb",)
    if outputs:
        cmd += ["--rendered-output-names", ",".join(outputs)]
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
    """Render a video along a saved camera path (ideal for 360° orbits)."""
    ensure_executable("ns-render")
    if not camera_path.exists():
        raise FileNotFoundError(f"Camera path file missing: {camera_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ns-render",
        "camera-path",
        "--load-config",
        str(config_path),
        "--camera-path-filename",
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
    """Render a video by interpolating between training cameras."""
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
