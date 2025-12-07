"""Run COLMAP SfM + MVS pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple

from ..utils import ensure_executable, run_command


def run_colmap_pipeline(
    images_dir: Path,
    workspace_dir: Path,
    *,
    matcher: str = "sequential",
    num_threads: int = 8,
    extra_args: Optional[Iterable[str]] = None,
    make_mesh: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """Run feature extraction, matching, mapping, MVS, fusion, and optional meshing."""
    ensure_executable("colmap")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    workspace_dir.mkdir(parents=True, exist_ok=True)
    db_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"
    dense_dir = workspace_dir / "dense"

    if db_path.exists():
        db_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    def _extend(cmd: list[str]) -> list[str]:
        if extra_args:
            cmd += list(extra_args)
        return cmd

    run_command(
        _extend(
            [
                "colmap",
                "feature_extractor",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--ImageReader.single_camera",
                "1",
                "--SiftExtraction.use_gpu",
                "1",
                "--SiftExtraction.num_threads",
                str(num_threads),
            ]
        )
    )

    if matcher == "exhaustive":
        match_cmd = [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            "1",
            "--SiftMatching.num_threads",
            str(num_threads),
        ]
    else:
        match_cmd = [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            "1",
            "--SiftMatching.num_threads",
            str(num_threads),
        ]
    run_command(_extend(match_cmd))

    run_command(
        _extend(
            [
                "colmap",
                "mapper",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_dir),
                "--Mapper.num_threads",
                str(num_threads),
            ]
        )
    )

    model_path = sparse_dir / "0"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No sparse model found at {model_path}. "
            "Did mapping fail?"
        )

    run_command(
        _extend(
            [
                "colmap",
                "image_undistorter",
                "--image_path",
                str(images_dir),
                "--input_path",
                str(model_path),
                "--output_path",
                str(dense_dir),
                "--output_type",
                "COLMAP",
            ]
        )
    )

    run_command(
        _extend(
            [
                "colmap",
                "patch_match_stereo",
                "--workspace_path",
                str(dense_dir),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
                "--PatchMatchStereo.num_threads",
                str(num_threads),
            ]
        )
    )

    fused_path = dense_dir / "fused.ply"
    run_command(
        _extend(
            [
                "colmap",
                "stereo_fusion",
                "--workspace_path",
                str(dense_dir),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--StereoFusion.num_threads",
                str(num_threads),
                "--output_path",
                str(fused_path),
            ]
        )
    )

    mesh_path: Optional[Path] = None
    if make_mesh:
        mesh_path = dense_dir / "mesh_poisson.ply"
        run_command(
            _extend(
                [
                    "colmap",
                    "poisson_mesher",
                    "--input_path",
                    str(fused_path),
                    "--output_path",
                    str(mesh_path),
                ]
            )
        )

    return fused_path, mesh_path


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(description="Run COLMAP SfM + MVS.")
    parser.add_argument("images_dir", type=Path, help="Directory of input images.")
    parser.add_argument("workspace_dir", type=Path, help="Output workspace directory.")
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive"],
        default="sequential",
        help="Matching strategy.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Threads for COLMAP stages.",
    )
    parser.add_argument(
        "--make-mesh",
        action="store_true",
        help="Run Poisson meshing after fusion.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Args appended to every COLMAP call.",
    )
    return parser


def _main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_colmap_pipeline(
        args.images_dir,
        args.workspace_dir,
        matcher=args.matcher,
        num_threads=args.num_threads,
        extra_args=args.extra_args,
        make_mesh=args.make_mesh,
    )


if __name__ == "__main__":
    _main()
