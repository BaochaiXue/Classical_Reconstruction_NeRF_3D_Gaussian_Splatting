"""Command-line entry point for the Nerf pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_pipeline_config
from .pipeline.export_assets import export_cameras, export_mesh
from .pipeline.render_views import (
    render_dataset_views,
    render_interpolate_video,
    render_orbit_video,
)
from .pipeline.train_nerf import train_nerf
from .pipeline.video_to_dataset import process_video_to_dataset
from .utils import find_latest_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video → Nerfstudio Nerf pipeline orchestrator."
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
        help="Skip ns-process-data (assumes dataset already exists).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip ns-train (assumes a finished run exists).",
    )
    parser.add_argument(
        "--skip-exports",
        action="store_true",
        help="Skip all rendering/export stages.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_pipeline_config(args.config)

    scene_name = cfg["scene_name"]
    video_path = Path(cfg["video_path"])
    dataset_dir = Path("data/processed") / scene_name

    if not args.skip_processing:
        process_video_to_dataset(
            video_path,
            dataset_dir,
            num_frames_target=cfg["processing"].get("num_frames_target"),
            extra_args=cfg["processing"].get("extra_args"),
        )

    if not args.skip_training:
        train_nerf(
            dataset_dir,
            scene_name=scene_name,
            method=cfg["training"]["method"],
            vis=cfg["training"]["vis"],
            extra_args=cfg["training"].get("extra_args"),
        )

    if args.skip_exports:
        return

    config_path = find_latest_config(scene_name, cfg["training"]["method"])

    figures_dir = Path("exports/figures") / scene_name
    videos_dir = Path("exports/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    cameras_dir = Path("exports/cameras") / scene_name
    meshes_dir = Path("exports/meshes") / scene_name

    render_cfg = cfg["render"]
    exports_cfg = cfg["exports"]

    if render_cfg.get("make_dataset_renders", False):
        render_dataset_views(
            config_path,
            figures_dir / "dataset",
            split=render_cfg.get("dataset_render_split", "train+test"),
            extra_args=render_cfg.get("extra_args"),
        )

    if exports_cfg.get("cameras", True):
        export_cameras(config_path, cameras_dir)

    mesh_cfg = exports_cfg.get("mesh", False)
    mesh_enabled = True
    mesh_args = None
    if isinstance(mesh_cfg, dict):
        mesh_enabled = mesh_cfg.get("enabled", True)
        mesh_args = mesh_cfg.get("extra_args")
    else:
        mesh_enabled = bool(mesh_cfg)
    if mesh_enabled:
        export_mesh(
            config_path,
            meshes_dir / scene_name,
            extra_args=mesh_args,
        )

    if render_cfg.get("make_orbit_video", False):
        camera_path = Path(render_cfg["camera_path_file"])
        if camera_path.exists():
            render_orbit_video(
                config_path,
                camera_path,
                videos_dir / f"{scene_name}_orbit.mp4",
                extra_args=render_cfg.get("extra_args"),
            )
        else:
            print(
                f"[warn] Camera path {camera_path} not found; skipping orbit render."
            )

    if render_cfg.get("make_interpolate_video", False):
        render_interpolate_video(
            config_path,
            videos_dir / f"{scene_name}_interpolate.mp4",
            extra_args=render_cfg.get("extra_args"),
        )


if __name__ == "__main__":
    main()
