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
        "--output-path",
        str(output_dir),
    ]
    run_command(cmd)


def export_mesh(
    config_path: Path,
    output_path: Path,
    *,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Export a mesh for the trained Nerf (if supported by the method)."""
    ensure_executable("ns-export")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-export",
        "mesh",
        "--load-config",
        str(config_path),
        "--output-path",
        str(output_path),
    ]
    if extra_args:
        cmd += list(extra_args)
    run_command(cmd)
