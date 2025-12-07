"""Command-line entry point for the classical reconstruction pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_pipeline_config
from .pipeline.export_assets import copy_outputs
from .pipeline.run_colmap import run_colmap_pipeline
from .pipeline.video_to_frames import extract_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video -> frames -> COLMAP SfM/MVS pipeline orchestrator."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Pipeline YAML config.",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip frame extraction (assumes images already exist).",
    )
    parser.add_argument(
        "--skip-colmap",
        action="store_true",
        help="Skip COLMAP (assumes outputs already exist).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_pipeline_config(args.config)

    scene_name = cfg["scene_name"]
    video_path = Path(cfg["video_path"])
    dataset_dir = Path("data/processed") / scene_name
    images_dir = dataset_dir / "images"
    workspace_dir = Path("outputs") / scene_name / "colmap"

    processing_cfg = cfg.get("processing", {})
    colmap_cfg = cfg.get("colmap", {})

    if not args.skip_processing:
        extract_frames(
            video_path,
            images_dir,
            fps=processing_cfg.get("frame_rate"),
            max_frames=processing_cfg.get("max_frames"),
        )

    fused_path = None
    mesh_path = None
    if not args.skip_colmap:
        fused_path, mesh_path = run_colmap_pipeline(
            images_dir,
            workspace_dir,
            matcher=colmap_cfg.get("matcher", "sequential"),
            num_threads=int(colmap_cfg.get("num_threads", 8)),
            extra_args=colmap_cfg.get("extra_args"),
            make_mesh=bool(colmap_cfg.get("make_mesh", False)),
        )

    if fused_path:
        copy_outputs(fused_path, mesh_path, scene_name=scene_name)


if __name__ == "__main__":
    main()
