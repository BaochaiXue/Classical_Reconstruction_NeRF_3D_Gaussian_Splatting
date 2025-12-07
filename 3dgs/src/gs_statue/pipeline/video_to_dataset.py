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
