"""Nerfstudio training wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..utils import ensure_executable, run_command


def train_nerf(
    dataset_dir: Path,
    *,
    scene_name: str,
    method: str = "nerfacto",
    vis: str = "viewer",
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Launch ns-train for the given dataset."""
    ensure_executable("ns-train")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    cmd = [
        "ns-train",
        method,
        "--data",
        str(dataset_dir),
        "--output-dir",
        "outputs",
        "--experiment-name",
        scene_name,
        "--vis",
        vis,
    ]
    if extra_args:
        cmd += list(extra_args)

    run_command(cmd)


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(description="Run ns-train for a dataset.")
    parser.add_argument("dataset", type=Path, help="Processed dataset directory.")
    parser.add_argument(
        "--scene",
        required=True,
        help="Experiment/scene name (used for outputs).",
    )
    parser.add_argument(
        "--method",
        default="nerfacto",
        help="Nerfstudio method (e.g., nerfacto).",
    )
    parser.add_argument(
        "--vis",
        default="viewer",
        help="Viewer mode (viewer, viewer+tensorboard, tensorboard).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to ns-train.",
    )
    return parser


def _main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    train_nerf(
        args.dataset,
        scene_name=args.scene,
        method=args.method,
        vis=args.vis,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    _main()
