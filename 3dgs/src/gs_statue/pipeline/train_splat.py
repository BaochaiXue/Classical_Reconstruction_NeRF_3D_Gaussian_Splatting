"""Nerfstudio training helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..utils import ensure_executable, run_command


def train_splat(
    data_dir: Path,
    *,
    method: str = "splatfacto",
    vis: str = "viewer+tensorboard",
    extra_args: Optional[Iterable[str]] = None,
    output_dir: Path | None = None,
) -> None:
    """Train a Nerfstudio 3DGS model via ns-train."""
    ensure_executable("ns-train")
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory missing: {data_dir}")

    cmd = [
        "ns-train",
        method,
        "--data",
        str(data_dir),
        "--vis",
        vis,
    ]
    if output_dir is not None:
        cmd += ["--output-dir", str(output_dir)]
    if extra_args:
        cmd += list(extra_args)

    run_command(cmd)
