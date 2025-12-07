"""Export utilities for COLMAP outputs."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


def copy_outputs(
    fused_path: Path,
    mesh_path: Optional[Path],
    *,
    scene_name: str,
) -> None:
    """Copy fused point cloud (and mesh if present) into exports/."""
    if fused_path.exists():
        dst_pc = Path("exports/pointclouds") / scene_name
        dst_pc.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fused_path, dst_pc / fused_path.name)
        print(f"[export] copied point cloud to {dst_pc/fused_path.name}")
    if mesh_path and mesh_path.exists():
        dst_mesh = Path("exports/meshes") / scene_name
        dst_mesh.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mesh_path, dst_mesh / mesh_path.name)
        print(f"[export] copied mesh to {dst_mesh/mesh_path.name}")
