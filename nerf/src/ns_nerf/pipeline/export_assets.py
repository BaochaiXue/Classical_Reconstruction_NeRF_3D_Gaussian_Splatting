"""Export helpers using ns-export."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..utils import ensure_executable, run_command


def export_cameras(config_path: Path, output_dir: Path) -> None:
    """Export camera parameters."""
    ensure_executable("ns-export")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-export",
        "cameras",
        "--load-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    run_command(cmd)


def export_mesh(
    config_path: Path,
    output_dir: Path,
    *,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Export a mesh for the trained Nerf (marching cubes)."""
    ensure_executable("ns-export")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-export",
        "marching-cubes",
        "--load-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if extra_args:
        cmd += list(extra_args)
    run_command(cmd)
