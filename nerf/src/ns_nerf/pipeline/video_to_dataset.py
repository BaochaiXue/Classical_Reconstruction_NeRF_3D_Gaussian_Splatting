"""Video -> Nerfstudio dataset utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..utils import ensure_executable, run_command


def process_video_to_dataset(
    video_path: Path,
    output_dir: Path,
    *,
    num_frames_target: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> Path:
    """Run ns-process-data on a video and return the dataset directory."""
    ensure_executable("ns-process-data")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ns-process-data",
        "video",
        "--data",
        str(video_path),
        "--output-dir",
        str(output_dir),
    ]
    if num_frames_target is not None:
        cmd += ["--num-frames-target", str(num_frames_target)]
    if extra_args:
        cmd += list(extra_args)

    run_command(cmd)
    return output_dir


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames + COLMAP from a video via ns-process-data."
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument("output", type=Path, help="Output dataset directory.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Target number of frames (ns-process-data downsampling).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to ns-process-data.",
    )
    return parser


def _main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    process_video_to_dataset(
        args.video,
        args.output,
        num_frames_target=args.num_frames,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    _main()
