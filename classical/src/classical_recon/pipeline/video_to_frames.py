"""Video -> frames using FFmpeg."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..utils import ensure_executable, run_command


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Path:
    """Extract frames from a video using ffmpeg."""
    ensure_executable("ffmpeg")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / "frame_%05d.jpg"
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]
    if max_frames is not None:
        cmd += ["-vframes", str(max_frames)]
    cmd += ["-qscale:v", "2", str(output_pattern)]

    run_command(cmd)
    return output_dir


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video", type=Path, help="Input video file.")
    parser.add_argument("output", type=Path, help="Output directory for frames.")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to save.",
    )
    return parser


def _main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    extract_frames(args.video, args.output, fps=args.fps, max_frames=args.max_frames)


if __name__ == "__main__":
    _main()
