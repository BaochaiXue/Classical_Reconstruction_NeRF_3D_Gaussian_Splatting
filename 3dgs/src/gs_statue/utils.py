"""Shared helpers for the statue pipeline."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Optional


def _inject_gsplat_include(env: dict) -> None:
    """Ensure gsplat's CUDA headers are on the include path for nvcc."""
    try:
        import gsplat

        include_dir = Path(gsplat.__file__).parent / "cuda" / "include"
    except Exception:
        return
    if include_dir.exists():
        existing = env.get("CPLUS_INCLUDE_PATH", "")
        if str(include_dir) not in existing.split(os.pathsep):
            env["CPLUS_INCLUDE_PATH"] = (
                f"{include_dir}{os.pathsep}{existing}" if existing else str(include_dir)
            )


class MissingDependencyError(RuntimeError):
    """Raised when an expected CLI tool is missing on PATH."""


def ensure_executable(name: str) -> None:
    """Ensure a CLI is available."""
    if shutil.which(name) is None:
        raise MissingDependencyError(
            f"Executable '{name}' not found on PATH. "
            "Double-check your Nerfstudio installation."
        )


def run_command(
    cmd: Iterable[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Run a command with logging."""
    printable = " ".join(str(part) for part in cmd)
    print(f"[run] {printable}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    _inject_gsplat_include(merged_env)
    subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        check=True,
    )


def find_latest_config(
    scene_name: str,
    method: str,
    *,
    outputs_root: Path | None = None,
) -> Path:
    """Locate the newest Nerfstudio run config for the given scene/method."""
    if outputs_root is None:
        outputs_root = Path("outputs")
    base = outputs_root / scene_name / method
    configs = sorted(base.glob("*/config.yml"))
    if not configs:
        raise FileNotFoundError(
            f"No Nerfstudio run found under {base}. "
            "Did you run ns-train yet?"
        )
    return configs[-1]
